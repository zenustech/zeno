#pragma once

#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "IOMat.h"
#include "TraceStuff.h"
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

template<bool MIS=true>
inline float3 bevel(MatInput&input, float radius=0.01f, uint sample=8, uint retry=0) {
    if (input.isShadowRay || radius == 0.0f || sample<=1) { return input.wldNorm; };

    bool inside = circleInsideTriangle(input.objPos, radius, input.vertices);
    if (inside) return input.wldNorm;

    uint32_t& seed = input.seed;

    Onb onb(input.objNorm);
    float3 axis[3] = {onb.m_normal, onb.m_binormal, onb.m_tangent};
    //float3 axis[3]{}; axis[0] = objNorm;
    //pbrt::CoordinateSystem(axis[0], axis[2], axis[3]);
    auto Sqr = [](float v) { return v * v; };
    
    // --- 3-bin accumulators ---
    uint16_t bins[3] {};
    float3   nrms[3] {};

    bool lost = false;
    int idx0 = 0, idx1 = 1, idx2 = 2;

    for (int i=0; i<sample; ++i) {
        float2 uu = { rnd(seed), rnd(seed) };
        auto offset = pbrt::SampleUniformDiskConcentric(uu);
        
        if (lost) {
            idx0 = (idx0 + 1) % 3;
        } else {
            auto prob = rnd(seed);
            idx0 = prob<0.5f? 0:(prob<0.75f? 1:2);
        }
        idx1 = (idx0 + 1) % 3;
        idx2 = (idx0 + 2) % 3;

        auto pos = input.objPos + radius * ( axis[idx0] + axis[idx1] * offset.x + axis[idx2] * offset.y);
        auto len2 = 1.0f - offset.x * offset.x  - offset.y * offset.y;
        auto len = radius * sqrtf(fmaxf(0.0f, len2));
        //if ( isnan(len) ) { len = 0.0f; }

        optixTraverse(input.gas, pos, -axis[idx0], fmaxf(0.0f, radius-len), radius+len, 0, EverythingMask,
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, 0);

        float3 hit_nrm = input.objNorm;

        if( optixHitObjectIsHit() ) {
            lost = false;

            const auto pid = optixHitObjectGetPrimitiveIndex();
            if (pid != input.priIdx) {
                float3 V[3] {};
                optixGetTriangleVertexData( input.gas, pid, input.sbtIdx,0, V );
                hit_nrm = normalize( cross(V[1]-V[0], V[2]-V[0]) );
            }
        } else {
            if (retry>0) { retry--; i--;}
            lost = true;
            continue;
        }

        // record hit
        bins[idx0]++;
        nrms[idx0]+=hit_nrm;
    }

    uint32_t count = bins[0] + bins[1] + bins[2];
    if (count == 0) return input.wldNorm;

    // --- compute empirical selection pdf ---
    float weight[3] = {0,0,0};
    weight[0] = float(bins[0]) / count;
    weight[1] = float(bins[1]) / count;
    weight[2] = float(bins[2]) / count;

    // --- compute average normal per axis ---
    for (char i=0; i<3; ++i) {
        nrms[i] = (bins[i] > 0) ? normalize(nrms[i] / bins[i]) : float3{};
    }
    if constexpr (MIS) {
        // --- MIS weight calculation using axis-averaged normals ---
        float pdf0 = weight[0] * fabsf(dot(axis[0], nrms[0]));
        float pdf1 = weight[1] * fabsf(dot(axis[1], nrms[1]));
        float pdf2 = weight[2] * fabsf(dot(axis[2], nrms[2]));

        float denom = Sqr(pdf0) + Sqr(pdf1) + Sqr(pdf2);
        if (denom > 1e-12f) {
            weight[0] = Sqr(pdf0) / denom;
            weight[1] = Sqr(pdf1) / denom;
            weight[2] = Sqr(pdf2) / denom;
        }
    }

    // final bevel normal
    float3 bevel_nrm = weight[0] * nrms[0] + weight[1] * nrms[1] + weight[2] * nrms[2];
    bevel_nrm = normalize(bevel_nrm);

    if (dot(bevel_nrm, input.objNorm) <= 0)
        bevel_nrm = normalize(bevel_nrm + input.objNorm);

    bevel_nrm = transformVector(bevel_nrm, input.worldToObject);
    return normalize(bevel_nrm);
}
