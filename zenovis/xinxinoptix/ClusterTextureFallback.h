#pragma once
#include "ClusterMaterialDevice.h"

namespace clusterMaterial {
// These small mip chains are uploaded before the first render and never
// evicted. They do not depend on a demand sampler or sparse page being ready.
__forceinline__ __device__ bool coarseFallback(
    const zeno_cluster_device::MaterialLaunch& launch,unsigned int id,
    float2 uv,float2 dx,float2 dy,float4& value)
{
    if(!launch.coarseTextures)return false;
    const auto coarse=launch.coarseTextures[id];
    if(!coarse.texture)return false;
    value=::tex2DLayeredGrad<float4>(coarse.texture,uv.x,uv.y,int(coarse.layer),dx,dy);
    return true;
}
}
