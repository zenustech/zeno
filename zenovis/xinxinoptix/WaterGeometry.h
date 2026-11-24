#pragma once
#include "WaterSampling.h"

// Geometry decoding for structured water clusters. Materials are evaluated by
// the same bound callable used by ordinary primitives.
namespace waterGeometry {
__forceinline__ __device__ const zeno_water_device::Launch* object(unsigned& id) {
    const auto slot=id>>zeno_water_device::ObjectClusterBits;
    id&=zeno_water_device::ObjectClusterSpan-1;
    return params.water.objects && slot<params.water.objectCount?params.water.objects+slot:nullptr;
}
__forceinline__ __device__ const zeno_water_device::Surface* surface() {
    auto id=optixGetClusterId();const auto* o=object(id);
    if(!o || !o->surfaces || id>=o->patchCount)return nullptr;
    const auto index=o->patches[id].surface;
    return index<o->surfaceCount?o->surfaces+index:nullptr;
}
__forceinline__ __device__ float3 normal(float3 position) {
    const auto* s=surface();if(!s)return make_float3(0,1,0);
    const auto local=s->transformed?zeno_water_device::transformPoint(s->geometryToObject,position):position;
    const auto wave=zeno_water_device::sample(*s,local.x,local.z);
    const auto n=make_float3(-wave.dx,1.f,-wave.dz);
    return normalize(s->transformed?zeno_water_device::transformNormal(s->geometryToObject,n):n);
}
__forceinline__ __device__ void attributes(MatInput& attrs,float2* uv) {
    const auto* s=surface();
    // U follows +X and V follows +Z on the undisplaced surface.
    const auto local=s && s->transformed?zeno_water_device::transformPoint(s->geometryToObject,attrs.objPos):attrs.objPos;
    const auto wave=s?zeno_water_device::sample(*s,local.x,local.z):zeno_water_device::Sample{};
    const auto localNormal=make_float3(-wave.dx,1.f,-wave.dz),localTangent=make_float3(1.f,wave.dx,0.f);
    const auto n=s && s->transformed?zeno_water_device::transformNormal(s->geometryToObject,localNormal):localNormal;
    attrs.N=normalize(transformNormal(n,attrs.worldToObject));
    const auto tangent=normalize(s && s->transformed?zeno_water_device::transformVector(s->objectToGeometry,localTangent):localTangent);
    attrs.T=normalize(transformVector(tangent,attrs.objectToWorld));
    attrs.B=cross(attrs.T,attrs.N);
    for(unsigned i=0;i<3;++i) {
        const auto p=s && s->transformed?zeno_water_device::transformPoint(s->geometryToObject,attrs.vertices[i]):attrs.vertices[i];
        uv[i]=s?make_float2((p.x-s->center[0])/s->size[0]+.5f,(p.z-s->center[2])/s->size[1]+.5f):make_float2(0.f);
    }
    attrs.proceduralTriangle=true;attrs.triangleUVs=uv;attrs.vertex_idx=make_uint3(0,1,2);
}
__forceinline__ __device__ float3 debug(unsigned int mode) {
    auto id=optixGetClusterId();const auto* o=object(id);
    if(!o || id>=o->patchCount)return make_float3(1,0,1);
    const auto patch=o->patches[id];
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
