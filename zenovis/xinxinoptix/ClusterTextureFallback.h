#pragma once
#include "ClusterMaterialDevice.h"

namespace clusterMaterial {
// These small mip chains are uploaded before the first render and never
// evicted. They do not depend on a demand sampler or sparse page being ready.
__forceinline__ __device__ bool coarseFallback(
    const zeno_cluster_device::MaterialLaunch& launch,unsigned int id,
    float2 uv,float2 dx,float2 dy,float4& value,unsigned int sampling=0,unsigned int width=1,unsigned int height=1)
{
    if(!launch.coarseTextures)return false;
    const auto coarse=launch.coarseTextures[id];
    if(!coarse.texture)return false;
    if(!sampling)value=::tex2DLayeredGrad<float4>(coarse.texture,uv.x,uv.y,int(coarse.layer),dx,dy);
    else {
        using namespace zeno_cluster_device;
        const float desired=getMipLevel(dx,dy,float(width),float(height),1.f);
        const auto wanted=textureSampleLevels(sampling,desired,31u);
        const unsigned int first=wanted.first>coarse.firstLevel?wanted.first-coarse.firstLevel:0;
        const unsigned int second=wanted.second>coarse.firstLevel?wanted.second-coarse.firstLevel:0;
        auto fetch=[&](unsigned int level) {
            const auto w=max(width>>(min(level+coarse.firstLevel,31u)),1u),h=max(height>>(min(level+coarse.firstLevel,31u)),1u);
            return ::tex2DLayeredLod<float4>(coarse.texture,textureSampleCoordinate(uv.x,w,wanted.point),
                textureSampleCoordinate(uv.y,h,wanted.point),int(coarse.layer),float(level));
        };
        value=fetch(first);
        if(wanted.blend>0.f&&first!=second)value=(1.f-wanted.blend)*value+wanted.blend*fetch(second);
    }
    return true;
}
}
