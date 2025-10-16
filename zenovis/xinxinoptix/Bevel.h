#pragma once

#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "IOMat.h"
#include "TraceStuff.h"

inline float3 bevel(MatInput&input, float radius=0.01f, uint sample_count=8) {
    if (input.isShadowRay || radius == 0.0f) { return input.wldNorm; };

    float3 bevel_nrm {};
    uint32_t& seed = input.seed;

    Onb onb(input.objNorm);
    float3 axis[3] = {onb.m_normal, onb.m_binormal, onb.m_tangent};

    //float3 axis[3]{}; axis[0] = objNorm;
    //pbrt::CoordinateSystem(axis[0], axis[2], axis[3]);
    int count = 0;

    bool lost = true;
    char idx0 = 0, idx1 = 1, idx2 = 2;
    
    for (int i=0; i<sample_count; ++i) {

        float2 uu = { rnd(seed), rnd(seed) };
        auto offset = pbrt::SampleUniformDiskConcentric(uu);

        if (lost) {
            idx0 = 3 * rnd(seed);
            idx1 = (idx0 + 1) % 3;
            idx2 = (idx0 + 2) % 3;
        }
        auto pos = input.objPos + radius * ( axis[idx0] + axis[idx1] * offset.x + axis[idx2] * offset.y);
        auto len2 = 1.0f - offset.x * offset.x  - offset.y * offset.y;
        auto len = radius * sqrtf(fmaxf(0.0f, len2));
        //if ( isnan(len) ) { len = 0.0f; }

        optixTraverse(input.gas, pos, -axis[idx0], fmaxf(0.0f, radius-len), radius+len, 0, EverythingMask,
                        OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, 0);

        if( optixHitObjectIsHit() ) {
            lost = false;
            count += 1;

            const auto pid = optixHitObjectGetPrimitiveIndex();
            if (pid == input.priIdx) {
                bevel_nrm += input.objNorm;
                continue; // use cached normal
            }
            float3 _vertices_[3];
            const float3& v0 = _vertices_[0];
            const float3& v1 = _vertices_[1];
            const float3& v2 = _vertices_[2];
            optixGetTriangleVertexData( input.gas, pid, input.sbtIdx,0, _vertices_ );
            bevel_nrm += normalize( cross( (v1-v0), (v2-v0) ) );
        } else {
            lost = true;
        }
    }
    if (count>0) {
        bevel_nrm /= count;
        bevel_nrm = normalize(bevel_nrm);
    } else {
        bevel_nrm = input.objNorm;
    }
    bevel_nrm = transformVector(bevel_nrm, input.worldToObject);
    return bevel_nrm;
}
