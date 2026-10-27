#pragma once
#include "SppmDevice.h"

// The material callable publishes a shadow request instead of tracing while
// the full material closure is live. Raygen resolves visibility afterward.
static __forceinline__ __device__ void sppmCameraMaterial(
    RadiancePRD& prd,MatOutput m,const MatInput& attrs,float offset,
    float3 direction,float distance,uint4 record)
{
    if (m.shadowReceiver>0.0f && prd.depth>=1u) {
        prd.direction=direction;prd.alphaHit=true;prd._tmin_=distance;return;
    }
    prd.mask_value=m.mask_value;prd.geometryNormal=attrs.wldNorm;
    if (m.doubleSide>.5f || m.thin>.5f)
        prd.geometryNormal=faceforward(prd.geometryNormal,-direction,prd.geometryNormal);
    prd.done=prd.alphaDepth>5 ? true : prd.done;
    if (m.opacity>rnd(prd.seed)) {
        prd.alphaHit=true;prd._tmin_=distance;
        prd.alphaDepth+=m.emissionOnly>.5f ? 1 : 0;return;
    }
    prd._tmax_=distance;
    if (m.isHair>.5f || m.subsurface>0 || sppmMax(float3(m.emission))>0) {
        atomicAdd(&params.sppm.statistics->unsupported,1ull);
        prd.radiance=make_float3(0);prd.done=true;return;
    }
    if (prd.depth==0u) {
        prd.record=record;
        if (prd.test_distance) { prd.done=true;prd.maxDistance=distance;return; }
    }
    m.roughness=clamp(m.roughness,0.f,1.f);
    if (prd.denoise) { prd.tmp_albedo=m.basecolor;prd.tmp_normal=m.nrm; }
    if (prd.diffDepth>=2) m.roughness=clamp(m.roughness,.3f,.99f);
    else if (prd.diffDepth>=1) m.roughness=clamp(m.roughness,.1f,.99f);
    if (prd.depth==0u && m.flatness>.5f) {
        prd.radiance=make_float3(0);prd.done=true;return;
    }
    if (m.emissionOnly>.5f) {
        float3 decay=make_float3(1);
        if (prd.curMatIdx>0u) {
            vec3 sigma,alpha;prd.readMat(sigma,alpha);
            decay=alpha.x<0 ? DisneyBSDF::Transmission(sigma,distance) :
                DisneyBSDF::Transmission2(sigma*alpha,sigma,prd.channelPDF,distance,true);
            prd.attenuation*=decay;
        }
        prd.radiance+=decay*float3(m.emission);prd.done=true;return;
    }
    float3 tangent=attrs.T,bitangent=attrs.B;
    if (tangent==float3{} || bitangent==float3{}) {
        Onb frame(m.nrm);tangent=frame.m_tangent;bitangent=frame.m_binormal;
    }
    sppmSurfaceMode<false,true>(prd,m,attrs.wldPos,prd.geometryNormal,m.nrm,tangent,bitangent,offset,direction,distance);
}
