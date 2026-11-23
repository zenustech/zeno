#pragma once
#include "WaterSampling.h"

namespace waterMaterial {
__forceinline__ __device__ const zeno_water_device::Surface* surface() {
    const auto id=optixGetClusterId();
    if(!params.water.surfaces || id>=params.water.patchCount)return nullptr;
    const auto index=params.water.patches[id].surface;
    return index<params.water.surfaceCount?params.water.surfaces+index:nullptr;
}
__forceinline__ __device__ float3 normal(float3 position) {
    const auto* s=surface();if(!s)return make_float3(0,1,0);
    const auto wave=zeno_water_device::sample(*s,position.x,position.z);
    return normalize(make_float3(-wave.dx,1.f,-wave.dz));
}
__forceinline__ __device__ MatOutput evaluate(const MatInput& attrs) {
    MatOutput out{};const auto* s=surface();
    const float ior=s?s->ior:1.333f,ratio=(ior-1.f)/(ior+1.f);
    out.basecolor=vec3(1.f);out.roughness=s?s->roughness:.025f;out.ior=ior;out.F0=ratio*ratio;
    out.specular=1.f;out.specTrans=1.f;out.smoothness=1.f;out.nrm=attrs.N;
    out.metalColor=vec3(1.f);out.transColor=vec3(1.f);out.transTint=vec3(1.f);out.reflectance=vec3(1.f);
    out.transTintDepth=1.f;out.transDistance=1.f;out.transScatterColor=vec3(1.f);
    out.importedFresnel=true;
    if(s)out.transTint=exp(-vec3(s->absorption[0],s->absorption[1],s->absorption[2]));
    return out;
}
__forceinline__ __device__ void shadow(ShadowPRD& prd) {
    const auto* s=surface();
    if(!s){prd.attanuation=vec3(0.f);optixTerminateRay();return;}
    const auto direction=normalize(optixGetObjectRayDirection());
    const auto position=optixGetObjectRayOrigin()+optixGetRayTmax()*optixGetObjectRayDirection();
    const float directionNormal=dot(direction,normal(position));
    const float cosine=fminf(1.f,fabsf(directionNormal));
    const float eta=directionNormal>0.f?s->ior:1.f/s->ior;
    if(eta*eta*(1.f-cosine*cosine)>1.f){prd.attanuation=vec3(0.f);optixTerminateRay();return;}
    const float f0=pbrt::Sqr((s->ior-1.f)/(s->ior+1.f));
    const float fresnel=f0+(1.f-f0)*powf(1.f-cosine,5.f);
    prd.attanuation*=1.f-fresnel;
    optixIgnoreIntersection();
}
__forceinline__ __device__ float3 debug(unsigned int mode) {
    const auto id=optixGetClusterId();
    if(id>=params.water.patchCount)return make_float3(1,0,1);
    const auto patch=params.water.patches[id];
    if(mode==zeno_cluster_device::Clusters) {
        unsigned int h=id*747796405u+2891336453u;h=((h>>((h>>28u)+4u))^h)*277803737u;h=(h>>22u)^h;
        return make_float3(.15f+.85f*(h&255u)/255.f,.15f+.85f*((h>>8)&255u)/255.f,.15f+.85f*((h>>16)&255u)/255.f);
    }
    if(mode==zeno_cluster_device::Lod) {
        const float t=fminf(float(zeno_water_device::MaxDepth-patch.depth)/5.f,4.f);
        return t<1?make_float3(0,t,1):t<2?make_float3(0,1,2-t):t<3?make_float3(t-2,1,0):make_float3(1,4-t,0);
    }
    float3 vertices[3];optixGetTriangleVertexData(vertices);
    const auto ng=normalize(optixTransformNormalFromObjectToWorldSpace(cross(vertices[1]-vertices[0],vertices[2]-vertices[0])));
    const auto p=optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin()+optixGetRayTmax()*optixGetWorldRayDirection());
    const auto ns=normalize(optixTransformNormalFromObjectToWorldSpace(normal(p)));
    if(mode==zeno_cluster_device::GeometricNormal)return .5f+.5f*ng;
    if(mode==zeno_cluster_device::ShadingNormal)return .5f+.5f*ns;
    if(mode==zeno_cluster_device::GeometricFacing)return dot(ng,optixGetWorldRayDirection())<0?make_float3(.05f,1,.05f):make_float3(1,.05f,.05f);
    if(mode==zeno_cluster_device::NormalAgreement){const float d=dot(ng,ns);return d>=0?make_float3(1-d,1,0):make_float3(1,1+d,0);}
    return make_float3(1.f);
}
}
