#pragma once
#include <OptiXToolkit/ShaderUtil/TextureUtil.h>
#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include "ClusterTextureFallback.h"
#include "TextureResidency.h"

namespace clusterMaterial {
using namespace zeno_cluster_device;
__forceinline__ __device__ void recordTextureImportance(unsigned int page,unsigned int weight,bool parent) {
    auto* table=params.cluster_materials.demandImportance;
    if(!table || page==InvalidTexturePage || page>=params.cluster_materials.textures.maxNumPages ||
       page<params.cluster_materials.textures.pageTable.capacity)return;
    unsigned int slot=(page*2654435761u)&(TextureDemandSlots-1);
    for(unsigned int probe=0;probe<8;++probe,slot=(slot+1)&(TextureDemandSlots-1)) {
        const auto old=atomicCAS(&table[slot].page,0u,page+1);
        if(!old || old==page+1){atomicAdd(parent?&table[slot].parent:&table[slot].fine,weight);return;}
    }
}
__forceinline__ __device__ bool doubleSided(const Material& m) {
    // Preserve authored alpha cutouts, blending and transmission. The opaque
    // override is a render policy; imported materials and caches stay intact.
    return (m.flags&DoubleSided) || (params.cluster_materials.twoSidedOpaque &&
        !(m.flags&(AlphaMask|AlphaBlend)) && m.transmission<=0.f);
}
struct Surface {
    const Material* material;
    float2 uv,dx,dy;
#if defined(ZENO_CLAS_TEXTURE_LOOKAHEAD)
    float2 aheadUv,aheadDx,aheadDy;
    bool ahead=false;
#endif
    float3 p[3];
    const ShadingVertex* a[3];
    const VertexNormal* vertices[3];
    const PlacementMaterials* placement;
    bool* textureMissing=nullptr;
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
__forceinline__ __device__ bool surface(Surface& s,float footprint,const float3* hitVertices=nullptr) {
    const auto& launch=params.cluster_materials;
    if(!launch.materials)return false;
    const auto c=params.cluster_data[optixGetClusterId()];
    const unsigned int triangle=optixGetPrimitiveIndex();
    if(!c.attributes) {
        atomicOr(launch.missing,2u);return false;
    }
    s.material=resolve(c,s.placement);if(!s.material)return false;
    const auto index=c.indices+triangle*3;
    float3 fetchedPositions[3];
    if(!hitVertices){optixGetTriangleVertexData(fetchedPositions);hitVertices=fetchedPositions;}
    for(int i=0;i<3;++i) {
        s.a[i]=c.attributes+index[i];s.vertices[i]=c.normals+index[i];
        s.p[i]=transform(*s.placement,hitVertices[i],true);
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
#if defined(ZENO_CLAS_TEXTURE_LOOKAHEAD)
        float3 aheadDelta{},aheadDx{},aheadDy{};
        const auto pixel=optixGetLaunchIndex();
        if(s.textureMissing&&launch.lookAheadReferences&&((pixel.x+pixel.y*params.width)&15u)==launch.lookAheadPhase) {
            const auto futureDirection=normalize(direction+dot(direction,params.cam.right)*launch.lookAheadRight+
                dot(direction,params.cam.up)*launch.lookAheadUp+dot(direction,params.cam.front)*launch.lookAheadFront);
            const auto futureEye=params.cam.eye+launch.lookAheadEye;
            const auto point=s.p[0]*(1-bary.x-bary.y)+s.p[1]*bary.x+s.p[2]*bary.y;
            const float divisor=dot(futureDirection,n);
            const float distance=dot(point-futureEye,n)/copysignf(fmaxf(fabsf(divisor),1.e-5f),divisor);
            if(fabsf(divisor)>.05f&&distance>0.f&&isfinite(distance)) {
                aheadDelta=futureEye+distance*futureDirection-point;
                const float ratio=distance/fmaxf(length(point-params.cam.eye),1.e-5f);
                const float futureFootprint=footprint*fminf(fmaxf(ratio,.5f),2.f)*launch.lookAheadSpread;
                Onb futureBasis(futureDirection);
                const float3 x=static_cast<float3>(futureBasis.m_tangent),y=static_cast<float3>(futureBasis.m_binormal);
                aheadDx=futureFootprint*(x-dot(x,n)/divisor*futureDirection);
                aheadDy=futureFootprint*(y-dot(y,n)/divisor*futureDirection);
                s.ahead=true;
            }
        }
#endif
        for(int k=0;k<2;++k) {
            const auto gradient=(s.a[1]->uv[k]-s.a[0]->uv[k])*a+(s.a[2]->uv[k]-s.a[0]->uv[k])*b;
            if(k==0){s.dx.x=dot(gradient,dx);s.dy.x=dot(gradient,dy);}
            else{s.dx.y=dot(gradient,dx);s.dy.y=dot(gradient,dy);}
#if defined(ZENO_CLAS_TEXTURE_LOOKAHEAD)
            if(s.ahead) {
                if(k==0){s.aheadUv.x=dot(gradient,aheadDelta);s.aheadDx.x=dot(gradient,aheadDx);s.aheadDy.x=dot(gradient,aheadDy);}
                else{s.aheadUv.y=dot(gradient,aheadDelta);s.aheadDx.y=dot(gradient,aheadDx);s.aheadDy.y=dot(gradient,aheadDy);}
            }
#endif
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
    if(s.textureMissing && params.cluster_materials.primaryCoverage) {
        const auto pixel=optixGetLaunchIndex();
        params.cluster_materials.primaryCoverage[pixel.x+pixel.y*params.width]=2u;
    }
    auto uv=uvTransform(b,s.uv),dx=uvTransform(b,s.dx,true),dy=uvTransform(b,s.dy,true);
    if(b.flip&1){uv.x=1-uv.x;dx.x=-dx.x;dy.x=-dy.x;}
    if(b.flip&2){uv.y=1-uv.y;dx.y=-dx.y;dy.y=-dy.y;}
    float4 value;
    bool permanent=false;
    // Wide secondary/shadow footprints often need only original mip levels
    // already resident in the dense arrays. Skip sparse paging for those
    // exact same levels. Primary requests keep their existing priority path.
    if(!s.textureMissing && params.cluster_materials.denseCoarseSampling && params.cluster_materials.coarseTextures) {
        const auto coarse=params.cluster_materials.coarseTextures[b.id];
        const auto scale=1u<<coarse.firstLevel;
        // Cropping odd dimensions or clamping a thin axis changes gradient
        // scale. Keep those textures on the normal sparse sampler.
        if(b.width>=scale && b.height>=scale && b.width%scale==0 && b.height%scale==0) {
            const float level=getMipLevel(dx,dy,b.width,b.height,1.f/8.f);
            // Leave one full mip of margin for the hardware LOD estimate and
            // trilinear neighbors; this never forces a coarser requested mip.
            if(level>=float(coarse.firstLevel)+1.f)
                permanent=coarseFallback(params.cluster_materials,unsigned(b.id),uv,dx,dy,value);
        }
    }
    if(!permanent) {
    bool resident=false;
    auto context=params.cluster_materials.textures;
    if(s.textureMissing && params.cluster_materials.primaryReferences)
        context.referenceBits=params.cluster_materials.primaryReferences;
    value=demandLoading::tex2DGrad<float4>(context,unsigned(b.id),uv.x,uv.y,dx,dy,&resident);
    bool samplerResident=false;
    const auto* sampler=reinterpret_cast<const demandLoading::TextureSampler*>(demandLoading::pagingMapOrRequest(context,unsigned(b.id),&samplerResident));
#if defined(ZENO_CLAS_TEXTURE_LOOKAHEAD)
    if(s.ahead&&samplerResident&&sampler&&sampler->desc.isSparseTexture) {
        auto step=uvTransform(b,s.aheadUv,true),fx=uvTransform(b,s.aheadDx,true),fy=uvTransform(b,s.aheadDy,true);
        if(b.flip&1){step.x=-step.x;fx.x=-fx.x;fy.x=-fy.x;}
        if(b.flip&2){step.y=-step.y;fx.y=-fx.y;fy.y=-fy.y;}
        const float level=getMipLevel(fx,fy,float(b.width),float(b.height),1.f/float(sampler->desc.maxAnisotropy));
        if(isfinite(level)&&isfinite(step.x)&&isfinite(step.y)) {
            const auto mip=unsigned(fminf(fmaxf(floorf(level),0.f),float(sampler->desc.numMipLevels-1)));
            const float x=float(1u<<sampler->desc.logTileWidth)/float(max(b.width>>mip,1u));
            const float y=float(1u<<sampler->desc.logTileHeight)/float(max(b.height>>mip,1u));
            step.x=fminf(fmaxf(step.x,-x),x);step.y=fminf(fmaxf(step.y,-y),y);
            demandLoading::requestTexFootprint2DGrad(*sampler,params.cluster_materials.lookAheadReferences,context.residenceBits,
                uv.x+step.x,uv.y+step.y,fx.x,fx.y,fy.x,fy.y,0.f,0.f);
        }
    }
#endif
    // Keep a coarse representation referenced while this texture is used,
    // including frames whose desired fine footprint is already resident.
    if(samplerResident && sampler && sampler->desc.isSparseTexture)
        demandLoading::pagingRequest(context.referenceBits,sampler->startPage);
    if(!resident) {
        if(s.textureMissing && samplerResident && sampler && params.cluster_materials.demandImportance) {
            const auto pixel=optixGetLaunchIndex();
            if(((pixel.x+pixel.y*17u+params.cluster_materials.residencyEpoch)&15u)==0) {
                const float desired=getMipLevel(dx,dy,float(b.width),float(b.height),1.f/float(sampler->desc.maxAnisotropy));
                if(isfinite(desired)) {
                    const unsigned int level=unsigned(fminf(fmaxf(floorf(desired),0.f),float(sampler->desc.numMipLevels-1)));
                    const unsigned int set=params.cluster_materials.materialSets && unsigned(slot)<3?s.material->feedbackSet[slot]:0u;
                    const unsigned int weight=set?4u:slot==BaseColor?4u:slot==Normal||slot==MetallicRoughness?2u:1u;
                    recordTextureImportance(textureCenterPage(*sampler,uv.x,uv.y,level),weight,false);
                    if(level+2<sampler->mipTailFirstLevel)
                        recordTextureImportance(textureCenterPage(*sampler,uv.x,uv.y,level+2),weight,true);
                    // Share coverage priority only across imported, proven
                    // compatible UV footprints. This requests the original
                    // mip for each map; it does not lower shading quality.
                    for(unsigned member=0;member<3 && set;++member) {
                        if(member==unsigned(slot) || !(set&(1u<<member)))continue;
                        const auto id=unsigned(s.material->textures[member].id);bool ready=false;
                        const auto* other=reinterpret_cast<const demandLoading::TextureSampler*>(demandLoading::pagingMapOrRequest(context,id,&ready));
                        if(!ready || !other || !other->desc.isSparseTexture)continue;
                        demandLoading::requestTexFootprint2DGrad(*other,context.referenceBits,context.residenceBits,uv.x,uv.y,dx.x,dx.y,dy.x,dy.y,0.f,0.f);
                        recordTextureImportance(textureCenterPage(*other,uv.x,uv.y,level),weight,false);
                        if(level+2<other->mipTailFirstLevel)
                            recordTextureImportance(textureCenterPage(*other,uv.x,uv.y,level+2),weight,true);
                    }
                }
            }
        }
        if(s.textureMissing)*s.textureMissing=true;
        // Bucket fallback flags to avoid contending on one device-wide word.
        // Reduce after rays, preserving the same miss and binding-error bits.
        const auto pixel=optixGetLaunchIndex();
        const auto bucket=(pixel.x+pixel.y*params.width)&params.cluster_materials.missingMask;
        const auto prior=atomicOr(params.cluster_materials.missing+bucket,1u);
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
            int first=isfinite(desired)?int(fminf(fmaxf(ceilf(desired),0.f),float(last))):last;
            if(params.cluster_materials.minMip)
                first=int(residentMipLowerBound(*sampler,context.residenceBits,uv.x,uv.y,unsigned(first),unsigned(last)));
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
__forceinline__ __device__ float3 tangent(const VertexNormal& v,unsigned int packed) {
    const auto n=make_float3(v.normal[0],v.normal[1],v.normal[2]);float3 t,b;
    if(n.z<-.99998796f){t=make_float3(0,-1,0);b=make_float3(-1,0,0);}
    else {const float a=1.f/(1.f+n.z),q=-n.x*n.y*a;t=make_float3(1-n.x*n.x*a,q,-n.x);b=make_float3(q,1-n.y*n.y*a,-n.y);}
    const float angle=(float((packed>>1)&511)/511.f*2.f-1.f)*M_PIf;
    return cosf(angle)*b+sinf(angle)*t;
}
__forceinline__ __device__ MatOutput evaluate(MatInput& attrs,float footprint,bool* textureMissing=nullptr) {
    MatOutput out{};
    out.basecolor=vec3(.5f);out.roughness=.7f;out.ior=1.5f;out.F0=.04f;out.doubleSide=1;out.smoothness=1;
    out.nrm=attrs.N;out.metalColor=vec3(1.f);out.transColor=vec3(1.f);out.transTint=vec3(1.f);out.reflectance=attrs.reflectance;
    Surface s{};s.textureMissing=textureMissing;
    if(!surface(s,footprint,params.cluster_materials.reuseTriangleVertices?attrs.vertices:nullptr))return out;
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
