#pragma once

#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "IOMat.h"
#include "TraceStuff.h"

inline float bevel_cubic_eval(const float radius, const float r)
{
    const float Rm = radius;
    if (r >= Rm) {
        return 0.0f;
    }

    /* integrate (2*pi*r * 10*(R - r)^3)/(pi * R^5) from 0 to R = 1 */
    const float Rm5 = (Rm * Rm) * (Rm * Rm) * Rm;
    const float f = Rm - r;
    const float num = f * f * f;

    return (10.0f * num) / (Rm5 * M_PIf);
}

inline float bevel_cubic_pdf(const float radius, const float r)
{
    return bevel_cubic_eval(radius, r);
}

/* solve 10x^2 - 20x^3 + 15x^4 - 4x^5 - xi == 0 */
inline float bevel_cubic_quintic_root_find(const float xi)
{
    /* newton-raphson iteration, usually succeeds in 2-4 iterations, except
    * outside 0.02 ... 0.98 where it can go up to 10, so overall performance
    * should not be too bad */
    const float tolerance = 1e-6f;
    const int max_iteration_count = 10;
    float x = 0.25f;
    int i;

    for (i = 0; i < max_iteration_count; i++) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float nx = (1.0f - x);

        const float f = 10.0f * x2 - 20.0f * x3 + 15.0f * x2 * x2 - 4.0f * x2 * x3 - xi;
        const float f_ = 20.0f * (x * nx) * (nx * nx);

        if (fabsf(f) < tolerance || f_ == 0.0f) {
            break;
        }

        x = clamp(x - f / f_, 0.f, 1.0f);
    }

    return x;
}

inline float2 sampleCubicDisk(float2& uu) {

    const float theta = 2.0f * M_PIf * uu.x;
    uu.y = bevel_cubic_quintic_root_find(uu.y);
    return uu.y * float2{cosf(theta), sinf(theta)};
}

inline float distancePointToLine(const float3& line_s, const float3& line_e, const float3& point) {

    float3 line = line_e - line_s;
    float3 diff = point - line_s;
    
    float proj = dot(line, diff) / dot(line, line);
    proj = clamp(proj, 0.0f, 1.0f);
    float3 vert = line_s + line * proj;
    auto tmp = point - vert;
    return length(tmp);
}

inline bool circleInsideTriangle(float3 center, float radius, float3* vertices) {
    if (nullptr == vertices) return true;
    #pragma unrolll
    for(char i=0; i<3; ++i) {
        auto X = vertices[i];
        auto Y = vertices[(i+1) % 3];
        auto t = distancePointToLine(X, Y, center);
        if (t < radius) return false;     
    }
    return true;
}

inline float3 bevel(MatInput&input, float radius, uint sample) {
    if (input.isShadowRay || radius == 0.0f || sample<=1) { return input.wldNorm; };

    bool inside = circleInsideTriangle(input.objPos, radius, input.vertices);
    if (inside) return input.wldNorm;

    uint32_t& seed = input.seed;
    seed = seed ^ 0x9e3779b9u;
    
    Onb onb(input.objNorm);
    float3 axis[3] = {onb.m_normal, onb.m_binormal, onb.m_tangent };
    //float3 axis[3]{}; axis[0] = objNorm;
    //pbrt::CoordinateSystem(axis[0], axis[2], axis[3]);
    auto Sqr = [](float v) { return v * v; };
    
    const float pick_pdf[3] = {0.50f, 0.25f, 0.25f};
    
    float3 bevel_nrm {};

    for (int i=0; i<sample; ++i) {

        int idx[3] = {0, 1, 2};
        float2 uu = { rnd(seed), rnd(seed) };

        auto prob = uu.x;
        float cdf = 0.0f;
        #pragma unrolll
        for(char j=0; j<3; ++j) {

            float tmp = cdf + pick_pdf[j];
            if (prob < tmp) {
                uu.x = (prob - cdf) / pick_pdf[j];
                idx[0] = j; idx[j] = 0; break;
            }
            cdf = tmp;
        }
        
        const auto& idx0=idx[0];
        const auto& idx1=idx[1];
        const auto& idx2=idx[2];

        auto offset = sampleCubicDisk(uu);

        auto pos = input.objPos + radius * ( axis[idx0] + axis[idx1] * offset.x + axis[idx2] * offset.y);
        auto len2 = 1.0f - offset.x * offset.x - offset.y * offset.y;
        auto len = radius * sqrtf(fmaxf(0.0f, len2));
        //if ( isnan(len) ) { len = 0.0f; }
        float tmin = radius-len;
        float tmax = radius+len;
        auto dir = -axis[idx0];

        do {
            optixTraverse(input.gas, pos, dir, fmaxf(0.0f, tmin), tmax, 0, EverythingMask,
                            OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, 0);

            float3 hit_nrm = input.objNorm;
            float3 hit_pos = input.objPos;

            if( optixHitObjectIsHit() ) {

                const auto pid = optixHitObjectGetPrimitiveIndex();
                if (pid != input.priIdx) {
                    float3 V[3] {};
                    optixGetTriangleVertexData( input.gas, pid, input.sbtIdx,0, V );
                    hit_nrm = normalize( cross(V[1]-V[0], V[2]-V[0]) );
                }
                tmin = optixHitObjectGetRayTmax();
                hit_pos = pos + tmin * (-axis[idx0]);
            
            } else {
                break;
            }

            const float di = length(hit_pos - input.objPos);
            const float Rd = bevel_cubic_pdf(radius, di);
            const float disk_pdf = bevel_cubic_pdf(1.0f, uu.y);

            const float pdf0 = pick_pdf[idx0] * fabsf(dot(axis[idx0], hit_nrm));
            const float pdf1 = pick_pdf[idx1] * fabsf(dot(axis[idx1], hit_nrm));
            const float pdf2 = pick_pdf[idx2] * fabsf(dot(axis[idx2], hit_nrm));

            float w =  (pdf0) / ( Sqr(pdf0) +  Sqr(pdf1) + Sqr(pdf2));
            w = w * Rd / disk_pdf;
            bevel_nrm += w * hit_nrm;

        } while (true);
    }

    if (dot(bevel_nrm, input.objNorm)<=0)
        bevel_nrm = bevel_nrm + input.objNorm;

    bevel_nrm = normalize(bevel_nrm);
    bevel_nrm = transformVector(bevel_nrm, input.worldToObject);
    return normalize(bevel_nrm);
}

template<bool TangentSpace=false>
inline float3 bevelCall(MatInput&input, float radius, uint sample, const float3& T, const float3& B, const float3& N) {
    auto bevel_nrm = bevel(input, radius, sample);

    if constexpr(TangentSpace) {

        auto t = dot(bevel_nrm, T);
        auto b = dot(bevel_nrm, B);
        auto n = dot(bevel_nrm, N);
        bevel_nrm = {t, b, n};
        bevel_nrm = normalize(bevel_nrm);
    }
    return bevel_nrm;
}