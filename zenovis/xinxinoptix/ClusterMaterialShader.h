#pragma once
#include <OptiXToolkit/ShaderUtil/TextureUtil.h>
#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include "ClusterTextureFallback.h"

namespace clusterMaterial {
using namespace zeno_cluster_device;
__forceinline__ __device__ bool doubleSided(const Material& m) {
    // Preserve authored alpha cutouts, blending and transmission. The opaque
    // override is a render policy; imported materials and caches stay intact.
    return (m.flags&DoubleSided) || (params.cluster_materials.twoSidedOpaque &&
        !(m.flags&(AlphaMask|AlphaBlend)) && m.transmission<=0.f);
}
struct Surface {
    const Material* material;
    float2 uv,dx,dy;
    float3 p[3];
    const ShadingVertex* a[3];
    const Vertex* vertices[3];
    const PlacementMaterials* placement;
};
__forceinline__ __device__ const Material* resolve(const Cluster& c,const PlacementMaterials*& placement) {
    const auto& launch=params.cluster_materials;
    const unsigned int p=optixGetInstanceId(),triangle=optixGetPrimitiveIndex();
    if(p>=launch.placementCount||triangle>=c.triangles||!c.materials){atomicOr(launch.missing,2u);return nullptr;}
    placement=launch.placements+p;const unsigned int slot=c.materials[triangle];
    if(slot>=placement->count){atomicOr(launch.missing,2u);return nullptr;}
    const auto id=launch.materialIds[placement->offset+slot];
    if(id>=launch.materialCount){atomicOr(launch.missing,2u);return nullptr;}
    return launch.materials+id;
}
__forceinline__ __device__ float3 transform(const PlacementMaterials& p,float3 v,bool point) {
    const auto* m=p.transform;
    return make_float3(m[0]*v.x+m[1]*v.y+m[2]*v.z+(point?m[3]:0.f),
                      m[4]*v.x+m[5]*v.y+m[6]*v.z+(point?m[7]:0.f),
                      m[8]*v.x+m[9]*v.y+m[10]*v.z+(point?m[11]:0.f));
}
__forceinline__ __device__ bool surface(Surface& s,float footprint) {
    const auto& launch=params.cluster_materials;
    if(!launch.materials)return false;
    const auto c=params.cluster_data[optixGetClusterId()];
    const unsigned int triangle=optixGetPrimitiveIndex();
    if(!c.attributes) {
        atomicOr(launch.missing,2u);return false;
    }
    s.material=resolve(c,s.placement);if(!s.material)return false;
    const auto index=c.indices+triangle*3;
    for(int i=0;i<3;++i) {
        s.a[i]=c.attributes+index[i];s.vertices[i]=c.vertices+index[i];
        const auto* v=s.vertices[i]->position;s.p[i]=transform(*s.placement,make_float3(v[0],v[1],v[2]),true);
    }
    const auto bary=optixGetTriangleBarycentrics();
    s.uv=make_float2(0.f);s.dx=s.dy=make_float2(0.f);
    for(int i=0;i<3;++i) {
        const float weight=i==0?1-bary.x-bary.y:i==1?bary.x:bary.y;
        s.uv.x+=weight*s.a[i]->uv[0];s.uv.y+=weight*s.a[i]->uv[1];
    }
    const auto e1=s.p[1]-s.p[0],e2=s.p[2]-s.p[0];auto n=cross(e1,e2);const float area=length(n);
    if(area>1.e-15f) {
        n=n/area;
        const auto direction=optixGetWorldRayDirection();
        Onb basis(direction);const float d=dot(direction,n);
        const float inverse=1.f/copysignf(fmaxf(fabsf(d),1.e-5f),d);
        const float3 axisX=static_cast<float3>(basis.m_tangent),axisY=static_cast<float3>(basis.m_binormal);
        const auto dx=footprint*(axisX-dot(axisX,n)*inverse*direction);
        const auto dy=footprint*(axisY-dot(axisY,n)*inverse*direction);
        const auto a=cross(e2,n)/area,b=cross(n,e1)/area;
        for(int k=0;k<2;++k) {
            const auto gradient=(s.a[1]->uv[k]-s.a[0]->uv[k])*a+(s.a[2]->uv[k]-s.a[0]->uv[k])*b;
            if(k==0){s.dx.x=dot(gradient,dx);s.dy.x=dot(gradient,dy);}
            else{s.dx.y=dot(gradient,dx);s.dy.y=dot(gradient,dy);}
        }
    }
    return true;
}
__forceinline__ __device__ float2 uvTransform(const TextureBinding& b,float2 uv,bool vector=false) {
    return make_float2(b.transform[0]*uv.x+b.transform[1]*uv.y+(vector?0.f:b.transform[2]),
                      b.transform[3]*uv.x+b.transform[4]*uv.y+(vector?0.f:b.transform[5]));
}
__forceinline__ __device__ float4 sample(const Surface& s,TextureSlot slot) {
    const auto& b=s.material->textures[slot];
    if(b.id<0)return make_float4(1.f);
    auto uv=uvTransform(b,s.uv),dx=uvTransform(b,s.dx,true),dy=uvTransform(b,s.dy,true);
    if(b.flip&1){uv.x=1-uv.x;dx.x=-dx.x;dy.x=-dy.x;}
    if(b.flip&2){uv.y=1-uv.y;dx.y=-dx.y;dy.y=-dy.y;}
    bool resident=false;
    const auto& context=params.cluster_materials.textures;
    auto value=demandLoading::tex2DGrad<float4>(context,unsigned(b.id),uv.x,uv.y,dx,dy,&resident);
    bool samplerResident=false;
    const auto* sampler=reinterpret_cast<const demandLoading::TextureSampler*>(demandLoading::pagingMapOrRequest(context,unsigned(b.id),&samplerResident));
    // Keep a coarse representation referenced while this texture is used,
    // including frames whose desired fine footprint is already resident.
    if(samplerResident && sampler && sampler->desc.isSparseTexture)
        demandLoading::pagingRequest(context.referenceBits,sampler->startPage);
    if(!resident) {
        const auto prior=atomicOr(params.cluster_materials.missing,1u);
        if(params.cluster_materials.diagnostics && !(prior&1u) && samplerResident && sampler) {
            bool actual=false;
            const auto raw=::tex2DGrad<float4>(sampler->texture,uv.x,uv.y,dx,dy,&actual);
            const bool mapped=demandLoading::requestTexFootprint2DGrad(*sampler,context.referenceBits,context.residenceBits,uv.x,uv.y,dx.x,dx.y,dy.x,dy.y,0.f,0.f);
            printf("[CLAS texture miss] id=%d uv=%.9g,%.9g dx=%.9g,%.9g dy=%.9g,%.9g hardware=%d table=%d tail=%u levels=%u value=%.9g,%.9g,%.9g,%.9g\n",b.id,uv.x,uv.y,dx.x,dx.y,dy.x,dy.y,int(actual),int(mapped),sampler->mipTailFirstLevel,sampler->desc.numMipLevels,raw.x,raw.y,raw.z,raw.w);
        }
        // Use the nearest resident coarser level rather than jumping to 4x4.
        // Probe without requesting every intermediate mip: the original fine
        // footprint and the tail above are the only missing pages we request.
        bool coarse=false;
        if(samplerResident && sampler) {
            const float desired=getMipLevel(dx,dy,float(b.width),float(b.height),1.f/float(sampler->desc.maxAnisotropy));
            const int last=int(sampler->desc.numMipLevels)-1;
            const int first=isfinite(desired)?int(fminf(fmaxf(ceilf(desired),0.f),float(last))):last;
            for(int level=first;level<=last;++level) {
                // Physical residency can precede the copy completing. Probe
                // the published table without admitting intermediate requests.
                if(sampler->desc.isSparseTexture && !demandLoading::requestTexFootprint2DLod(*sampler,nullptr,context.residenceBits,uv.x,uv.y,float(level)))continue;
                value=::tex2DLod<float4>(sampler->texture,uv.x,uv.y,float(level),&coarse);
                if(coarse) {
                    demandLoading::requestTexFootprint2DLod(*sampler,context.referenceBits,context.residenceBits,uv.x,uv.y,float(level));
                    break;
                }
            }
        }
        if(!coarse)coarse=coarseFallback(params.cluster_materials,unsigned(b.id),uv,dx,dy,value);
        // An unseen emissive map must not invent a white light while cold.
        if(!coarse)return slot==Normal?make_float4(.5f,.5f,1.f,1.f):slot==Emissive?make_float4(0.f):make_float4(1.f);
    }
    const float components[6]={value.x,value.y,value.z,value.w,0.f,1.f};
    return make_float4(components[b.swizzle&255],components[(b.swizzle>>8)&255],components[(b.swizzle>>16)&255],components[b.swizzle>>24]);
}
__forceinline__ __device__ float alpha(const Surface& s,float4 base) {
    const auto& m=*s.material;
    if(!doubleSided(m)&&optixIsBackFaceHit())return 0.f;
    const float a=fminf(fmaxf(base.w*m.baseColor[3],0.f),1.f);
    if(m.flags&AlphaMask)return a>=m.alphaCutoff?1.f:0.f;
    return m.flags&AlphaBlend?a:1.f;
}
__forceinline__ __device__ float shadowCoverage(float footprint) {
    const auto c=params.cluster_data[optixGetClusterId()];const PlacementMaterials* placement=nullptr;
    const auto* material=resolve(c,placement);if(!material)return 1.f;
    if(!doubleSided(*material)&&optixIsBackFaceHit())return 0.f;
    if(!(material->flags&(AlphaMask|AlphaBlend)))return 1.f;
    Surface s{};return surface(s,footprint)?alpha(s,sample(s,BaseColor)):1.f;
}
__forceinline__ __device__ float3 tangent(const Vertex& v,unsigned int packed) {
    const auto n=make_float3(v.normal[0],v.normal[1],v.normal[2]);float3 t,b;
    if(n.z<-.99998796f){t=make_float3(0,-1,0);b=make_float3(-1,0,0);}
    else {const float a=1.f/(1.f+n.z),q=-n.x*n.y*a;t=make_float3(1-n.x*n.x*a,q,-n.x);b=make_float3(q,1-n.y*n.y*a,-n.y);}
    const float angle=(float((packed>>1)&511)/511.f*2.f-1.f)*M_PIf;
    return cosf(angle)*b+sinf(angle)*t;
}
__forceinline__ __device__ MatOutput evaluate(MatInput& attrs,float footprint) {
    MatOutput out{};
    out.basecolor=vec3(.5f);out.roughness=.7f;out.ior=1.5f;out.F0=.04f;out.doubleSide=1;out.smoothness=1;
    out.nrm=attrs.N;out.metalColor=vec3(1.f);out.transColor=vec3(1.f);out.transTint=vec3(1.f);out.reflectance=attrs.reflectance;
    Surface s{};if(!surface(s,footprint))return out;
    const auto& m=*s.material;
    if(!doubleSided(m)&&optixIsBackFaceHit()){out.opacity=1.f;return out;}
    const auto base=sample(s,BaseColor);
    out.basecolor=vec3(base.x,base.y,base.z)*vec3(m.baseColor[0],m.baseColor[1],m.baseColor[2]);
    out.opacity=1-alpha(s,base);out.doubleSide=doubleSided(m)?1.f:0.f;
    if(out.opacity>=1.f)return out;
    const auto mr=sample(s,MetallicRoughness),spec=sample(s,Specular);
    out.metallic=fminf(fmaxf(m.metallic*mr.z,0.f),1.f);out.roughness=fminf(fmaxf(m.roughness*mr.y,0.f),1.f);
    out.specular=m.specular*spec.w;
    out.ior=m.ior;out.F0=pbrt::Sqr((m.ior-1.f)/(m.ior+1.f));
    out.specTrans=m.transmission*sample(s,Transmission).x;out.thin=out.specTrans>0?1.f:0.f;
    out.transColor=out.basecolor;out.transTint=out.basecolor;
    const auto emission=sample(s,Emissive);
    out.emission=vec3(emission.x,emission.y,emission.z)*vec3(m.emission[0],m.emission[1],m.emission[2]);
    if(m.flags&Unlit){out.emission+=out.basecolor;out.emissionOnly=1.f;}
    if(m.textures[Normal].id>=0) {
        const auto& binding=m.textures[Normal];
        const auto uv0=uvTransform(binding,make_float2(s.a[0]->uv[0],s.a[0]->uv[1]));
        const auto uv1=uvTransform(binding,make_float2(s.a[1]->uv[0],s.a[1]->uv[1]));
        const auto uv2=uvTransform(binding,make_float2(s.a[2]->uv[0],s.a[2]->uv[1]));
        auto n=static_cast<float3>(attrs.N),t=make_float3(0.f),b=make_float3(0.f);
        const bool authored=(s.a[0]->tangent&s.a[1]->tangent&s.a[2]->tangent&0x80000000u)&&
            binding.transform[0]==1&&binding.transform[1]==0&&binding.transform[3]==0&&binding.transform[4]==1;
        if(authored) {
            const auto bary=optixGetTriangleBarycentrics();
            t=interp(bary,tangent(*s.vertices[0],s.a[0]->tangent),tangent(*s.vertices[1],s.a[1]->tangent),tangent(*s.vertices[2],s.a[2]->tangent));
            t=transform(*s.placement,t,false);t=t-dot(t,n)*n;
            const auto c0=transform(*s.placement,make_float3(1,0,0),false),c1=transform(*s.placement,make_float3(0,1,0),false),c2=transform(*s.placement,make_float3(0,0,1),false);
            const float sign=(s.a[0]->tangent&1)?1.f:-1.f;
            b=cross(n,t)*sign*copysignf(1.f,dot(cross(c0,c1),c2));
        } else {
            const auto d1=uv1-uv0,d2=uv2-uv0;const float determinant=d1.x*d2.y-d1.y*d2.x;
            if(fabsf(determinant)>1.e-15f){t=((s.p[1]-s.p[0])*d2.y-(s.p[2]-s.p[0])*d1.y)/determinant;b=((s.p[2]-s.p[0])*d1.x-(s.p[1]-s.p[0])*d2.x)/determinant;}
        }
        if(length(t)>1.e-10f&&length(b)>1.e-10f) {
            t=normalize(t-dot(t,n)*n);const float sign=dot(cross(n,t),b)<0?-1.f:1.f;b=sign*cross(n,t);
            const auto tex=sample(s,Normal);const float rawX=tex.x*2-1,rawY=tex.y*2-1;
            const float x=rawX*m.normalScale,y=rawY*m.normalScale;
            const float z=sqrtf(fmaxf(0.f,1.f-rawX*rawX-rawY*rawY));
            const auto mapped=x*t+y*b+z*n;
            if(length(mapped)>1.e-10f)out.nrm=normalize(mapped);
            attrs.T=t;attrs.B=b;
        }
    }
    return out;
}
}
