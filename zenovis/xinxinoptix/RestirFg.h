#pragma once
#include "SppmDevice.h"

// ReSTIR-FG reconnection with pairwise generalized balance MIS. The stored
// photon direction permits full BSDF reevaluation at the secondary vertex.
static __forceinline__ __device__ float fgTarget(float3 v) {
    return sppmFinite(v) ? fmaxf(0.0f,dot(v,make_float3(.2126f,.7152f,.0722f))) : 0.0f;
}
// A shared leaf avoids duplicating this closure at every reconnection/trace
// site. Inlining it inflated continuation spills and the reuse instruction set.
static __noinline__ __device__ float3 fgBsdf(const FgSurface& s, float3 wi, float3 wo, RadiancePRD& prd) {
    if (fgLambertian(s.material) && dot(s.shadingNormal,wo)>0.0f) {
        const float cosine=dot(s.shadingNormal,wi);
        return fgDiffuseBsdf(s.material.basecolor,cosine);
    }
    float pdf=0;
    // fgGatherable excludes transmission, and unsupported closures fall back
    // before a surface is published. Reuse therefore needs only opaque BSDFs.
    return DisneyBSDF::EvaluateOpaqueSppm(s.material,wi,wo,s.tangent,s.bitangent,s.shadingNormal,pdf);
}
static __forceinline__ __device__ FgQuery fgQuery(const FgSurface& s, bool caustic, uint32_t& seed, RadiancePRD& prd) {
    FgQuery q={}; q.surface=&s; q.seed=seed; q.causticOnly=caustic;
    q.diffuse=!caustic && fgLambertian(s.material) && dot(s.shadingNormal,s.wo)>0.0f;
    if (params.sppm.searchHandle) {
        unsigned lo,hi; packPointer(&q,lo,hi);
        optixTrace(params.sppm.searchHandle,s.position,make_float3(1,0,0),0.0f,
            fmaxf(params.sppm.initialRadius*1e-4f,1e-30f),0.0f,caustic ? 2u : 3u,OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,2,lo,hi);
    }
    sppmStatAdd(&params.sppm.statistics->gathered,q.gathered);
    sppmStatAdd(&params.sppm.statistics->caustic,q.caustics);
    seed=q.seed; return q;
}
static __forceinline__ __device__ float3 fgValue(const FgSurface& s, const FgReservoir& r,
    float& geometry, RadiancePRD& prd, bool visibility) {
    geometry=0;
    if (!s.valid || !r.valid) return make_float3(0);
    float3 delta=r.point.position-s.position;
    const float d2=lengthSquared(delta);
    if (!(d2>1e-12f)) return make_float3(0);
    const float3 wi=delta*rsqrtf(d2);
    geometry=fabsf(dot(r.point.normal,-wi))/d2;
    if (!(geometry>0)) return make_float3(0);
    float3 outgoing;
    if (r.diffuse) {
        if (!(dot(r.point.shadingNormal,-wi)>0.0f)) return make_float3(0);
        outgoing=r.diffuseRadiance;
    } else {
        const float cosine=fabsf(dot(r.point.shadingNormal,r.photonDirection));
        if (!(cosine>1e-6f)) return make_float3(0);
        outgoing=r.incident*fgBsdf(r.point,r.photonDirection,-wi,prd)/cosine;
    }
    float3 value=fgBsdf(s,wi,s.wo,prd)*outgoing;
    value*=float3(DisneyBSDF::Transmission(vec3(s.extinction),sqrtf(d2)));
    if (!(fgTarget(value)>0)) return make_float3(0);
    if (visibility) {
        const float3 a=rtgems::offset_ray(s.position,dot(s.normal,wi)>=0?s.normal:-s.normal);
        const float3 b=rtgems::offset_ray(r.point.position,dot(r.point.normal,-wi)>=0?r.point.normal:-r.point.normal);
        delta=b-a;
        const float distance=length(delta);
        if (!(distance>0)) return make_float3(0);
        ShadowPRD shadow={}; shadow.seed=prd.seed; shadow.depth=s.depth;
        shadow.origin=a; shadow.ShadowNormal=dot(s.normal,wi)>=0?s.normal:-s.normal;
        shadow.oppositeOrigin=rtgems::offset_ray(s.position,-shadow.ShadowNormal);
        shadow.attanuation=make_float3(1);
        traceOcclusion(params.handle,a,delta/distance,0,nextafterf(distance,0.0f),&shadow,EverythingMask^VolumeMatMask);
        prd.seed=shadow.seed;
        if (shadow.nonThinTransHit) return make_float3(0);
        value*=shadow.attanuation;
    }
    return value;
}
static __forceinline__ __device__ float3 fgCausticValue(const FgSurface& s, const FgCausticReservoir& r, RadiancePRD& prd) {
    const auto& p=r.photon;
    if (!s.valid || !r.valid || dot(s.normal,p.normal)<=0 ||
        lengthSquared(s.position-p.position)>params.sppm.initialRadius*params.sppm.initialRadius) return make_float3(0);
    const float cosine=fabsf(dot(s.shadingNormal,p.wi));
    return cosine>1e-6f ? p.power*fgBsdf(s,p.wi,s.wo,prd)/cosine : make_float3(0);
}
static __forceinline__ __device__ float fgShiftTarget(const FgSurface& destination,
    const FgSurface& source,const FgReservoir& r,float& geometry,RadiancePRD& prd,float3& value) {
    geometry=0;
    value=make_float3(0);
    if (!r.valid || !(r.jacobian>0)) return 0;
    // Define a symmetric partial reconnection domain. Very short connections
    // relative to the pixel displacement are unstable near corners. Excluding
    // a shift gives its original sample full MIS weight; no energy is clipped.
    const float separation=lengthSquared(destination.position-source.position);
    if (fminf(lengthSquared(r.point.position-source.position),
              lengthSquared(r.point.position-destination.position))<=4*separation) return 0;
    const float3 delta=r.point.position-destination.position;
    const float d2=lengthSquared(delta);
    if (!(d2>1e-12f)) return 0;
    geometry=fabsf(dot(r.point.normal,-delta*rsqrtf(d2)))/d2;
    // The bounds are powers of two: this is the same domain test without a
    // double-precision division at every shifted connection.
    const double originalGeometry=r.jacobian;
    if (double(geometry)<originalGeometry*(1.0/16.0) || double(geometry)>originalGeometry*16.0) {
        geometry=0;return 0;
    }
    value=fgValue(destination,r,geometry,prd,true);
    return fgTarget(value);
}
static __forceinline__ __device__ float3 fgEstimate(const FgReservoirPair& a,RadiancePRD& prd) {
    float geometry;
    return fgValue(a.source,a.fg,geometry,prd,false)*a.fg.weight+
        fgCausticValue(a.source,a.caustic,prd)*a.caustic.weight;
}
static __forceinline__ __device__ void fgCombine(FgReservoirPair& a,const FgReservoirPair& b,RadiancePRD& prd,
    float3* estimate=nullptr, const float3* currentFg=nullptr, const float3* currentCaustic=nullptr) {
    if (!b.source.valid || dot(a.source.normal,b.source.normal)<.8f || a.source.depth!=b.source.depth) return;
    auto& x=a.fg; const auto& y=b.fg;
    float gx=0,gy=0;
    float3 vx,vy;
    const float tx=fgShiftTarget(b.source,a.source,x,gx,prd,vx);
    const float ty=fgShiftTarget(a.source,b.source,y,gy,prd,vy);
    const double jx=x.jacobian>0 ? double(gx)/x.jacobian : 0;
    const double jy=y.jacobian>0 ? double(gy)/y.jacobian : 0;
    const auto weights=fgPairWeights(x.target,y.target,tx,ty,jx,jy,x.weight,y.weight,x.confidence,y.confidence);
    const double wx=weights.current, wy=weights.other;
    auto& c=a.caustic; const auto& d=b.caustic;
    const float tc=fgTarget(fgCausticValue(b.source,c,prd));
    const float3 vd=fgCausticValue(a.source,d,prd);
    const float td=fgTarget(vd);
    const auto causticWeights=fgPairWeights(c.target,d.target,tc,td,1,1,c.weight,d.weight,c.confidence,d.confidence);
    const double wc=causticWeights.current, wd=causticWeights.other;
    if (estimate) {
        // Integrate out the last reservoir-selection random variable. Each
        // pair remains the same generalized-MIS estimator, but its two RGB
        // contributions are evaluated directly. No light or weight is capped.
        float geometry;
        *estimate=make_float3(0);
        if(x.target>0) *estimate+=(currentFg ? *currentFg : fgValue(a.source,x,geometry,prd,false))*float((wx)/x.target);
        if(ty>0) *estimate+=vy*float((wy)/ty);
        if(c.target>0) *estimate+=(currentCaustic ? *currentCaustic : fgCausticValue(a.source,c,prd))*float((wc)/c.target);
        if(td>0) *estimate+=vd*float((wd)/td);
        return;
    }
    const uint32_t confidence=min(params.sppm.fgConfidence,x.confidence+y.confidence);
    if (wy>0 && double(rnd(prd.seed))*(wx+wy)<wy) { x=y; x.target=ty; x.jacobian=gy; }
    x.weight=x.target>0 ? float((wx+wy)/x.target) : 0;
    x.confidence=confidence;

    const uint32_t cc=min(params.sppm.fgConfidence,c.confidence+d.confidence);
    if (wd>0 && double(rnd(prd.seed))*(wc+wd)<wd) { c=d; c.target=td; }
    c.weight=c.target>0 ? float((wc+wd)/c.target) : 0;
    c.confidence=cc;
}
extern "C" __global__ void __raygen__fg_initial() {
    const auto index=optixGetLaunchIndex(); const uint32_t i=index.y*params.width+index.x;
    auto& out=params.sppm.fgInitial[i];
    out.fg.confidence=out.caustic.confidence=1u;
    out.fg.valid=out.caustic.valid=0u;
    out.fg.target=out.fg.weight=out.fg.jacobian=0.0f;
    out.caustic.target=out.caustic.weight=0.0f;
    if (!out.source.valid) return;
    RadiancePRD prd={}; prd.sppmMode=1u;
    prd.seed=pcg_hash(tea<4>(i,params.sppm.iteration)^params.outside_random_number^0x61d75a3bu);
    const double area=double(params.sppm.photonsPerIteration)*M_PIf*params.sppm.initialRadius*params.sppm.initialRadius;
    auto q=fgQuery(out.source,true,prd.seed,prd);
    if (q.sum>0) {
        out.caustic.photon=q.selected; out.caustic.valid=1;
        out.caustic.target=q.selectedTarget;
        out.caustic.weight=float(q.sum/(q.selectedTarget*area));
    }
    const auto& candidate=params.sppm.fgCandidates[i];
    if (candidate.point.valid && candidate.pdf>0) {
        q=fgQuery(candidate.point,false,prd.seed,prd);
        if (q.diffuse ? fgTarget(q.diffuseRadiance)>0 : q.sum>0) {
            auto& r=out.fg; r.point=candidate.point; r.valid=1;
            r.diffuse=q.diffuse;
            if (r.diffuse) r.diffuseRadiance=q.diffuseRadiance/float(area);
            else {
                r.incident=q.selected.power*float(q.sum/(q.selectedTarget*area));
                r.photonDirection=q.selected.wi;
            }
            r.target=fgTarget(fgValue(out.source,r,r.jacobian,prd,false));
            r.weight=1.0f/candidate.pdf;
        }
    }
    if (out.fg.target>0 || out.caustic.target>0) sppmStatAdd(&params.sppm.statistics->fgInitial);
}
extern "C" __global__ void __raygen__fg_resample() {
    const auto index=optixGetLaunchIndex(); const uint32_t i=index.y*params.width+index.x;
    auto current=params.sppm.fgInitial[i];
    RadiancePRD prd={}; prd.sppmMode=1u;
    prd.seed=pcg_hash(tea<4>(i,params.sppm.iteration)^params.outside_random_number^0x0b564e39u);
    float3 spatialEstimate=make_float3(0);
    unsigned spatialCount=0;
    if (current.source.valid && params.sppm.fgReuse) {
        const auto& temporal=params.sppm.fgPrevious[i];
        if (params.sppm.iteration && temporal.source.valid) {
            fgCombine(current,temporal,prd);
            sppmStatAdd(&params.sppm.statistics->fgTemporal);
        }
        // Keep temporal history before spatial resampling. Feeding a shared
        // spatial winner back into every neighbor amplifies sample correlation.
        params.sppm.fgCurrent[i]=current;
        // Pairwise output never changes current. Evaluate its two original
        // contributions once, instead of up to twice for every neighbor.
        float3 currentFg=make_float3(0), currentCaustic=make_float3(0);
        if (params.sppm.fgPairwiseOutput) {
            float geometry;
            currentFg=fgValue(current.source,current.fg,geometry,prd,false);
            currentCaustic=fgCausticValue(current.source,current.caustic,prd);
        }
        for (unsigned k=0;k<params.sppm.fgSpatialSamples;++k) {
            const float angle=2*M_PIf*rnd(prd.seed), radius=8*sqrtf(rnd(prd.seed));
            const int x=int(index.x)+int(roundf(radius*cosf(angle))), y=int(index.y)+int(roundf(radius*sinf(angle)));
            if (x<0 || y<0 || x>=int(params.width) || y>=int(params.height) || (x==index.x && y==index.y)) continue;
            const auto& neighbor=params.sppm.fgInitial[y*params.width+x];
            if (neighbor.source.valid) {
                if(params.sppm.fgPairwiseOutput) {
                    float3 pair=currentFg*current.fg.weight+currentCaustic*current.caustic.weight;
                    fgCombine(current,neighbor,prd,&pair,&currentFg,&currentCaustic);
                    spatialEstimate+=pair;
                    ++spatialCount;
                } else fgCombine(current,neighbor,prd);
                sppmStatAdd(&params.sppm.statistics->fgSpatial);
            }
        }
    } else params.sppm.fgCurrent[i]=current;
    const float3 value=spatialCount ? spatialEstimate/float(spatialCount) : fgEstimate(current,prd);
    auto& pixel=params.sppm.pixels[i];
    if (current.source.valid) pixel.tau+=pixel.beta*value;
}
