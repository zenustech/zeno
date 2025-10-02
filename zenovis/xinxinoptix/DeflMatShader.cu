#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>

#include <sutil/vec_math.h>
#include "optixPathTracer.h"

#include "TraceStuff.h"
#include "zxxglslvec.h"

#include "IOMat.h"
#include "Light.h"

#include "DisneyBRDF.h"
#include "DisneyBSDF.h"

#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>

#ifndef __CUDACC_RTC__
#define _P_TYPE_ 0
#endif

#if (_P_TYPE_==2)
#include "Curves.h"
#endif

__inline__ __device__ bool isBadVector(const vec3& vector) {

    bool bad = !isfinite(vector[0]) || !isfinite(vector[1]) || !isfinite(vector[2]);
    return bad? true : lengthSquared(vector) == 0.0f;
}

__inline__ __device__ bool isBadVector(const float3& vector) {
    return isBadVector(reinterpret_cast<const vec3&>(vector));
}

__inline__ __device__ bool isfinite(const float3& vector) {
    return isfinite(vector.x) && isfinite(vector.y) && isfinite(vector.z);
}

extern "C" __global__ void __anyhit__shadow_cutout()
{
    auto rt_data = (HitGroupData*)optixGetSbtDataPointer();
    auto dc_index = rt_data->dc_index;

    auto prd = getPRD<ShadowPRD>();
    prd->radiance = make_float3(0,0,0);
    bool opaque = rt_data->opacity == +1.0f;
    bool useomm = rt_data->opacity == -1.0f;
    
    auto skip = opaque;
    if (useomm) {
        skip |= prd->depth<=1 && rt_data->binaryShadowTestDirectRay;
        skip |= prd->depth>=2 && rt_data->binaryShadowTestIndirectRay;
    }
    if ( skip ) {
        prd->attanuation = {};
        optixTerminateRay();
        return;
    }

    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint               primIdx = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    MatInput attrs {};
    attrs.ptype = optixGetPrimitiveType();
    attrs.gas = gas;
    attrs.priIdx = primIdx;
    attrs.sbtIdx = sbtGASIndex;
    attrs.instId = optixGetInstanceId();
    attrs.instIdx = optixGetInstanceIndex();
    attrs.rayLength = optixGetRayTmax();
    attrs.isBackFace = optixIsBackFaceHit();
    attrs.seed = prd->seed;

    float3& objPos = attrs.objPos; 
    float3& objNorm = attrs.objNorm; 
    float3& wldPos = attrs.wldPos; 
    float3& wldNorm = attrs.wldNorm; 

    float3 shadingNorm;

    optixGetObjectToWorldTransformMatrix((float*)attrs.objectToWorld);
    optixGetWorldToObjectTransformMatrix((float*)attrs.worldToObject);

#if (_P_TYPE_==2)
    prd->attanuation = vec3(0);
    optixTerminateRay();
    return;
#elif (_P_TYPE_==1)
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, primIdx, sbtGASIndex, 0.f, &q );

    wldPos = P;
    objPos = optixTransformPointFromWorldToObjectSpace( wldPos );

    float3& _center_object_ = *(float3*)&q; 

    objNorm  = ( objPos - _center_object_ ) / q.w;
    wldNorm  = normalize( optixTransformNormalFromObjectToWorldSpace( objNorm ) );

    auto _origin_object_ = optixGetObjectRayOrigin();
    bool sphere_external_ray = length(_origin_object_ - _center_object_) > q.w;

    wldNorm = faceforward( wldNorm, -ray_dir, wldNorm );
#else

    float3 _vertices_[3];
    attrs.vertices = _vertices_;
    const float3& v0 = _vertices_[0];
    const float3& v1 = _vertices_[1];
    const float3& v2 = _vertices_[2];
    optixGetTriangleVertexData( gas, primIdx, sbtGASIndex, 0, _vertices_);

    float2 barys = optixGetTriangleBarycentrics();
    objPos = interp(barys, v0, v1, v2);
    objNorm = normalize(cross(v1-v0, v2-v0));

    wldPos = optixTransformPointFromObjectToWorldSpace(objPos);
    wldNorm = optixTransformNormalFromObjectToWorldSpace(objNorm);
    wldNorm = normalize(wldNorm);

    let gas_ptr = (void**)optixGetGASPointerFromHandle(gas);
    let idx_ptr = reinterpret_cast<uint3*>(  *(gas_ptr-1) );
    attrs.vertex_idx = idx_ptr[primIdx];

    attrs.barys2 = barys;
    attrs.N = reinterpret_cast<TriangleInput&>(attrs).interpNorm();
    attrs.T = reinterpret_cast<TriangleInput&>(attrs).interpTang();
#endif

    attrs.isShadowRay = true;
    mat4 World2Object(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    attrs.World2ObjectMat = (float*)&World2Object;
    optixGetWorldToObjectTransformMatrix(attrs.World2ObjectMat);
    attrs.World2ObjectMat[15] = 1.0f;
    MatOutput mats = optixDirectCall<MatOutput, cudaTextureObject_t[], MatInput&>( dc_index, rt_data->textures, attrs );
    shadingNorm = mats.nrm;
    shadingNorm = faceforward( shadingNorm, -ray_dir, shadingNorm );
    
    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01f,0.99f);

    auto opacity = mats.opacity;
    auto specTrans = mats.specTrans;
    auto ior = mats.ior;
    auto thin = mats.thin;

    //opacity = clamp(opacity, 0.0f, 0.99f);
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99f) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
        return;
    }
    else
    {
        //roll a dice
        float p = rnd(prd->seed);

        float skip = opacity;
        #if (_P_TYPE_==1)
            if (sphere_external_ray) {
                skip *= opacity;
            }
        #endif

        if (p < skip){
            optixIgnoreIntersection();
            return;
        }else{
          if(mats.isHair>0.5f)
          {
             vec3 extinction = exp( - DisneyBSDF::CalculateExtinction(mats.sssParam,1.0f) );
             if(p<min(min(extinction.x, extinction.y), extinction.z))
             {
               optixIgnoreIntersection();
               return;
             }
          }

            if(length(prd->attanuation) < 0.01f){
                prd->attanuation = vec3(0.0f);
                optixTerminateRay();
                return;
            }

            if(specTrans==0.0f){
                prd->attanuation = vec3(0.0f);
                optixTerminateRay();
                return;
            }
            
            if(specTrans > 0.0f){

                if(thin == 0.0f && ior>=1.0f)
                {
                    prd->nonThinTransHit++;
                }
                if(rnd(prd->seed)<(1-specTrans)||prd->nonThinTransHit>1)
                {
                    prd->attanuation = vec3(0,0,0);
                    optixTerminateRay();
                    return;
                }

                float nDi = fabs(dot(shadingNorm, normalize(ray_dir)));
                vec3 fakeTrans = vec3(1)-BRDFBasics::fresnelSchlick(vec3(1) - mats.transColor,nDi);
                prd->attanuation = prd->attanuation * fakeTrans;

                #if (_P_TYPE_==1)
                    if (sphere_external_ray) {
                        prd->attanuation *= vec3(1, 0, 0);
                        if (nDi < (1.0f-_FLT_EPL_)) {
                            prd->attanuation = {};
                            optixTerminateRay(); return;
                        } else {
                            prd->attanuation *= fakeTrans;
                        }
                    }
                #endif
                optixIgnoreIntersection();
                return;
            }
        }

        prd->attanuation = vec3(0);
        optixTerminateRay();
        return;
    }
}

static __inline__ __device__
vec3 projectedBarycentricCoord(vec3 p, vec3 q, vec3 u, vec3 v)
{
    vec3 n = cross(u,v);
    float a = 1.0f / dot(n,n);
    vec3 w = p - q;
    vec3 o;
    o.z = dot(cross(u,w),n) * a;
    o.y = dot(cross(w,v),n) * a;
    o.x = 1.0 - o.y - o.z;
    return o;
}

static __inline__ __device__
vec3 bezierOffset(vec3 P, vec3 A, vec3 B, vec3 C, vec3 nA, vec3 nB, vec3 nC, vec3 uvw)
{
    vec3 tmpu = P - A, tmpv = P - B, tmpw = P - C;
    float dotu = min(0.0, dot(tmpu, nA));
    float dotv = min(0.0, dot(tmpv, nB));
    float dotw = min(0.0, dot(tmpw, nC));
    tmpu = tmpu - dotu*nA;
    tmpv = tmpv - dotv*nB;
    tmpw = tmpw - dotw*nC;
    return uvw.x*tmpu + uvw.y*tmpv + uvw.z*tmpw;
}

static __forceinline__ __device__
vec3 FMA(vec3 a, vec3 b, vec3 c) {
    return {
        fmaf(a.x, b.x, c.x),
        fmaf(a.y, b.y, c.y),
        fmaf(a.z, b.z, c.z)
    };
    //return a * b + c;
};

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();
    prd->radiance = make_float3(0,0,0);
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint               primIdx = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    let rt_data = (HitGroupData*)optixGetSbtDataPointer();

    MatInput attrs {};
    attrs.ptype = optixGetPrimitiveType();
    attrs.gas = gas;
    attrs.priIdx = primIdx;
    attrs.sbtIdx = sbtGASIndex;
    attrs.instId = optixGetInstanceId();
    attrs.instIdx = optixGetInstanceIndex();
    attrs.rayLength = optixGetRayTmax();
    attrs.isBackFace = optixIsBackFaceHit();
    attrs.seed = prd->seed;

    float3 bezierOff {};
    auto dc_index = rt_data->dc_index;
    
    float3& objPos = attrs.objPos; 
    float3& objNorm = attrs.objNorm; 
    float3& wldPos = attrs.wldPos; 
    float3& wldNorm = attrs.wldNorm; 

    float objOffset; float wldOffset;
    float3 shadingNorm;
    
    optixGetObjectToWorldTransformMatrix((float*)attrs.objectToWorld);
    optixGetWorldToObjectTransformMatrix((float*)attrs.worldToObject);

    const float c0 = 5.9604644775390625E-8f;
    const float c1 = 1.788139769587360206060111522674560546875E-7f;
    const float c2 = 1.19209317972490680404007434844970703125E-7f;

#if (_P_TYPE_==2)

    auto curveAttr = CurveAttributes( optixGetPrimitiveType(), primIdx );
    objNorm = curveAttr.normal;
    // bound object space error due to reconstruction and intersection
    vec3 objErr = FMA( vec3( c0 ), abs( curveAttr.center ), vec3( c1 * curveAttr.radius ) );
    objOffset = dot( objErr, abs( objNorm ) );
    SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );

    if (isBadVector(objNorm)) {
        prd->done = true;
        return;
    }

    attrs.N = wldNorm;
    attrs.T = normalize( optixTransformVectorFromObjectToWorldSpace(curveAttr.tangent) );
    assert( dot(attrs.N, attrs.T) );
    attrs.B = cross(attrs.T, attrs.N);

    auto gas_ptr = (char*)optixGetGASPointerFromHandle(gas);
    auto& aux = *(CurveGroupAux*)(gas_ptr-sizeof(CurveGroupAux));

    uint strandIndex = aux.strand_i[primIdx].x;

    float  segmentU   = optixGetCurveParameter();
    float2 strand_u = aux.strand_u[primIdx];
    float u = strand_u.x + segmentU * strand_u.y;
    //attrs.uv = {u, (float)strandIndex/ aux.strand_info.count, 0};

#elif (_P_TYPE_==1)

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, primIdx, sbtGASIndex, 0.0f, &q );
    float3& sphere_center = *(float3*)&q;
    objPos   = optixTransformPointFromWorldToObjectSpace(P);
    objNorm  = normalize( ( objPos - sphere_center ) / q.w );

    objPos = sphere_center + objNorm * q.w;

    vec3 objErr = FMA( vec3( c0 ), abs( sphere_center ), vec3( c1 * q.w ) );
    objOffset = dot( objErr, abs( objNorm ) );

    SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );
    attrs.N = wldNorm;
#else
    
    float3 _vertices_[3];
    attrs.vertices = _vertices_;
    const float3& v0 = _vertices_[0];
    const float3& v1 = _vertices_[1];
    const float3& v2 = _vertices_[2];
    optixGetTriangleVertexData(gas, primIdx, sbtGASIndex, 0, _vertices_);

    const float2 barys = optixGetTriangleBarycentrics();
    SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, v0, v1, v2, barys );
    SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );
    
    if (isBadVector(wldNorm)) 
    {  
        prd->done = true;
        return;
    }

    let gas_ptr = (void**)optixGetGASPointerFromHandle(gas);
    let idx_ptr = reinterpret_cast<uint3*>(  *(gas_ptr-1) );
    attrs.vertex_idx = idx_ptr[primIdx];

    uint16_t* mat_ptr = reinterpret_cast<uint16_t*>(*(gas_ptr-6) );
    if ((uint64_t)mat_ptr != 0) {
        dc_index = mat_ptr[primIdx];
    }

    attrs.barys2 = barys;
    attrs.N = reinterpret_cast<const TriangleInput&>(attrs).interpNorm();
    attrs.T = reinterpret_cast<const TriangleInput&>(attrs).interpTang();

#endif

    float3 CUR_TOTAL_TRANS = make_float3(1,1,1);
    if(float3{} == attrs.T) {
        Onb a(attrs.N);
        attrs.T = a.m_tangent;
        attrs.B = a.m_binormal;
    } else {
        attrs.B = cross(attrs.T, attrs.N);
    }

    attrs.V = -(ray_dir);
    attrs.isShadowRay = false;
    mat4 World2Object(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
    attrs.World2ObjectMat = (float*)&World2Object;
    optixGetWorldToObjectTransformMatrix(attrs.World2ObjectMat);
    attrs.World2ObjectMat[15] = 1.0f;
    MatOutput mats = optixDirectCall<MatOutput, cudaTextureObject_t[], MatInput&>(rt_data->dc_index , rt_data->textures, attrs );
    prd->mask_value = mats.mask_value;
    prd->geometryNormal = attrs.wldNorm;
    bool geoNormalFlipped = false;
    if(mats.doubleSide>0.5f || mats.thin>0.5f) {
        auto before = prd->geometryNormal;
        //mats.nrm = faceforward( mats.nrm, attrs.V, mats.nrm );
        prd->geometryNormal  = faceforward( prd->geometryNormal , -ray_dir, prd->geometryNormal  );
        //geoNormalFlipped = dot(before, prd->geometryNormal)<0;
    }
    prd->done = prd->alphaDepth>5?true:prd->done;
    if( mats.opacity > rnd(prd->seed)) { // it's actually transparency not opacity
        prd->alphaHit = true;
        float travel_dist = optixGetRayTmax() - prd->_tmin_;
        prd->_tmin_ = optixGetRayTmax();
        prd->alphaDepth += mats.emissionOnly>0.5?1:0;



        prd->origin = prd->origin;
        if (prd->test_distance) return; 
        
//        if (prd->curMatIdx > 0) {
//            vec3 sigma_t, ss_alpha;
//            prd->readMat(sigma_t, ss_alpha);
//            if (ss_alpha.x < 0.0f) { // is inside Glass
//                auto decay = DisneyBSDF::Transmission(sigma_t, travel_dist);
//                prd->attenuation *= decay;
//                CUR_TOTAL_TRANS  *= decay;
//            } else {
//                auto decay = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF,travel_dist, true);
//                prd->attenuation *= decay;
//                CUR_TOTAL_TRANS  *= decay;
//            }
//        }
        return;
    }
    prd->_tmax_ = optixGetRayTmax();

    if (prd->depth==0) {
        *reinterpret_cast<uint64_t*>(&prd->record.x) = gas;
        prd->record.z = dc_index;
        prd->record.w = primIdx;

        if (prd->test_distance) {
            prd->done = true;
            prd->maxDistance = optixGetRayTmax();
            return;
        }
    }
    shadingNorm = geoNormalFlipped?-mats.nrm:mats.nrm;

#if (_P_TYPE_!=0)
    mats.smoothness = 0;
#else
    if (mats.smoothness > 0 && mats.shadowTerminatorOffset > 0) {

        auto barys3 = vec3(1-barys.x-barys.y, barys.x, barys.y);
        let nrm_ptr = reinterpret_cast<ushort3*>(*(gas_ptr-4) );
        float c = geoNormalFlipped?-1.0f:1.0f;
        float3 n0 = normalize( decodeHalf(nrm_ptr[ attrs.vertex_idx.x ]) );
        float3 n1 = normalize( decodeHalf(nrm_ptr[ attrs.vertex_idx.y ]) );
        float3 n2 = normalize( decodeHalf(nrm_ptr[ attrs.vertex_idx.z ]) );

        const auto localpos = (1-barys.x-barys.y)*v0 + barys.x * v1 + barys.y * v2;
        const auto offset = bezierOffset(localpos, v0, v1, v2, n0, n1, n2, barys3);
        bezierOff = float3(offset);
        const auto local_len = length(bezierOff);

        if (local_len > 0) {

            auto tmp = optixTransformNormalFromObjectToWorldSpace(bezierOff);
            auto len = local_len/length(tmp); len = len * len;
            bezierOff = mats.shadowTerminatorOffset * len * tmp;
            if (!isfinite(bezierOff)) bezierOff = {};
        }
    }
#endif

    mats.roughness = clamp(mats.roughness, 0.00f,1.0f);

    if (prd->denoise) {

        if(0.0f == mats.roughness) {
            prd->tmp_albedo = make_float3(1.0f);
        } else {
            prd->tmp_albedo = mats.basecolor;
        }
        prd->tmp_normal = shadingNorm;
    }
//    mats.subsurface = prd->depth>2?0:mats.subsurface;
    bool next_ray_is_going_inside = false;
    //mats.sssParam = mats.subsurface>0 ? mats.subsurface*mats.sssParam : mats.sssParam;
    //mats.subsurface = mats.subsurface>0 ? 1 : 0;

    /* MODME */
    if(prd->diffDepth>=2)
        mats.roughness = clamp(mats.roughness, 0.3,0.99);
    else if(prd->diffDepth>=1)
        mats.roughness = clamp(mats.roughness, 0.2,0.99);

    if(prd->isSS == true) {
//        if(prd->print_info)
//    {
//        printf("hit and from sss,depth:%d; Attenuation_in : %f,%f,%f; \n ", prd->depth,prd->attenuation.x, prd->attenuation.y, prd->attenuation.z);
//    }
        //mats.basecolor = vec3(1.0f);
        mats.roughness = max(mats.roughness , 0.1f);
        mats.anisotropic = 0.0f;
        mats.sheen = 0.0f;
        mats.clearcoat = 0.0f;
        mats.specTrans = 0.0f;
        mats.specular = 0.0f;
        //mats.ior = 1.0f;
        if(mats.subsurface==0.0f){
            prd->samplePdf = 1.0f;
            prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
            prd->readMat(prd->sigma_t, prd->ss_alpha);
            auto trans = DisneyBSDF::Transmission2(prd->sigma_s(), prd->sigma_t, prd->channelPDF, optixGetRayTmax(), true);
            prd->attenuation *= trans;
            CUR_TOTAL_TRANS  *= trans;
            //prd->origin = P;
            prd->direction = ray_dir;
            prd->_tmin_ = optixGetRayTmax();
            prd->done = true;
            return;
        }
        if(mats.subsurface>0.0f && dot(normalize(ray_dir), shadingNorm)<0.0f){
            prd->attenuation = make_float3(0.0f,0.0f,0.0f);
            CUR_TOTAL_TRANS  = make_float3(0.0f,0.0f,0.0f);
            prd->radiance = make_float3(0.0f,0.0f,0.0f);
            prd->done = true;
            return;
        }
    }

    prd->countEmitted = false;

    if(prd->depth==0&&mats.flatness>0.5)
    {
        prd->radiance = make_float3(0.0f);
        prd->done = true;
        return;
    }
    if(mats.emissionOnly > 0.5f){
        if (prd->curMatIdx > 0) {
          vec3 sigma_t, ss_alpha;
          //vec3 sigma_t, ss_alpha;
          prd->readMat(sigma_t, ss_alpha);
          if (ss_alpha.x < 0.0f) { // is inside Glass
            auto decay = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
            prd->attenuation  *= decay;
            CUR_TOTAL_TRANS   *= decay;
          } else {
            auto decay = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
            prd->attenuation *= decay;
            CUR_TOTAL_TRANS  *= decay;
          }
        }
        prd->radiance += CUR_TOTAL_TRANS * mats.emission;
        prd->done = true;
        return;
    }
//    if(prd->print_info)
//    {
//        printf("hit and stopped,depth:%d; Attenuation_in : %f,%f,%f; \n ", prd->depth,prd->attenuation.x, prd->attenuation.y, prd->attenuation.z);
//    }
    float is_refl;
    float3 inDir = ray_dir;
    vec3 wi = vec3(0.0f);
    float pdf = 0.0f;
    float rPdf = 0.0f;
    float fPdf = 0.0f;
    float rrPdf = 0.0f;

    float3 T = geoNormalFlipped?-attrs.T:attrs.T;
    float3 B = attrs.B;
    float3 N = shadingNorm;

    if (float3{}==T || float3{}==B)
    {
        Onb a(N);
        T = a.m_tangent;
        B = a.m_binormal;
    }

    DisneyBSDF::SurfaceEventFlags flag;
    DisneyBSDF::PhaseFunctions phaseFuncion;
    vec3 extinction;
    vec3 reflectance = vec3(0.0f);
    bool isDiff = false;
    bool isSS = false;
    bool isTrans = false;
    flag = DisneyBSDF::scatterEvent;

    if(prd->depth>1 && mats.roughness>0.4) mats.specular = 0.0f;
    while(DisneyBSDF::SampleDisney3(
                prd->seed,
                prd->eventseed,
                mats,
                T,
                B,
                N,
                prd->geometryNormal,
                -normalize(ray_dir),
                mats.thin>0.5f,
                next_ray_is_going_inside,
                wi,
                reflectance,
                rPdf,
                fPdf,
                flag,
                prd->medium,
                extinction,
                isDiff,
                isSS,
                isTrans,
                prd->minSpecRough
                )  == false)
        {
            isSS = false;
            isDiff = false;
            prd->samplePdf = fPdf;
            reflectance = fPdf>0?(reflectance/fPdf):vec3(0.0f);
            prd->done = fPdf>0?true:prd->done;
            flag = DisneyBSDF::scatterEvent;
        }

    if (isBadVector(wi)) {
        prd->done = true;
        prd->depth += 1;
        return;
    }
        
    prd->samplePdf = fPdf;
    reflectance = fPdf>0?(reflectance/fPdf):vec3(0.0f);
    prd->done = fPdf>0?prd->done:true;
    prd->isSS = isSS;
    pdf = 1.0;
    if(isDiff || prd->diffDepth>0){
        prd->diffDepth++;
    }
//    if(prd->depth>=3 && prd->hit_type==DIFFUSE_HIT)
//        prd->done = true;
//    if(prd->print_info)
//    {
//        printf("after sample,depth:%d; reflectance : %f,%f,%f; \n ", prd->depth,reflectance.x, reflectance.y, reflectance.z);
//    }
    bool coming_out_from_sss = false;
    bool going_in_to_sss = false;
    
    bool inToOut = false;
    bool outToIn = false;

    bool istransmission = dot(vec3(prd->geometryNormal), vec3(wi)) * dot(vec3(prd->geometryNormal), vec3(-normalize(ray_dir)))<0;
    //istransmission = (istransmission && thin<0.5 && mats.doubleSide==false);
    if(istransmission || flag == DisneyBSDF::diracEvent) {
    //if(flag == DisneyBSDF::transmissionEvent || flag == DisneyBSDF::diracEvent) {
        next_ray_is_going_inside = dot(vec3(prd->geometryNormal),vec3(wi))<=0;
    }
    coming_out_from_sss =  ((mats.thin<0.5f) && mats.subsurface>0 && isSS==false && istransmission);

    prd->max_depth = ((prd->depth==0 && isSS) || (prd->depth>0 && (mats.specTrans>0||mats.isHair>0)) )?32:prd->max_depth;

    if (isSS && mats.thin>0.5 && prd->curMatIdx==0)
    {
        isSS = false; // thin SSS
        //prd->max_depth = 4;
    }



    if(mats.thin>0.5f || mats.doubleSide>0.5f)
    {
        if (prd->curMatIdx > 0) {
            vec3 sigma_t, ss_alpha;
            prd->readMat(sigma_t, ss_alpha);

            vec3 trans;
            if (ss_alpha.x<0.0f) { // is inside Glass
                trans = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
            } else {
                trans = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
            }
            prd->attenuation *= trans;
            CUR_TOTAL_TRANS  *= trans;
        }

        next_ray_is_going_inside = false;

    }else{
    
        //if(flag == DisneyBSDF::transmissionEvent || flag == DisneyBSDF::diracEvent) {
        if(istransmission || flag == DisneyBSDF::diracEvent) {
            if(next_ray_is_going_inside){
//    if(prd->print_info)
//    {
//        printf("hit going in,depth:%d; Attenuation : %f,%f,%f; \n ", prd->depth,prd->attenuation.x, prd->attenuation.y, prd->attenuation.z);
//    }
                    outToIn = true;
                    inToOut = false;

                    prd->medium = DisneyBSDF::PhaseFunctions::isotropic;

                    if (prd->curMatIdx > 0) {
                        vec3 sigma_t, ss_alpha;
                        //vec3 sigma_t, ss_alpha;0
                        prd->readMat(sigma_t, ss_alpha);
                        if (ss_alpha.x < 0.0f) { // is inside Glass
                            auto decay = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                            prd->attenuation  *= decay;
                            CUR_TOTAL_TRANS   *= decay;
                        } else {
                            auto decay = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
                            prd->attenuation  *= decay;
                            CUR_TOTAL_TRANS   *= decay;
                        }
                    }
                    prd->channelPDF = vec3(1.0f/3.0f);
                    if (isTrans) {
                        vec3 channelPDF = vec3(1.0f/3.0f);
                        prd->pushMat(extinction);
                        prd->isSS = false;
                        prd->scatterDistance = mats.scatterDistance;
                        prd->maxDistance = mats.scatterStep>0.5f? DisneyBSDF::SampleDistance(prd->seed, prd->scatterDistance) : 1e16f;
                    } else {
                        prd->sssAttenBegin = prd->attenuation;
                        prd->sssDirBegin = ray_dir;

                        prd->attenuation *= vec3(1.0f);
                        float min_alpha = 0.2f;
                        if(prd->ss_alpha.x<min_alpha)
                        {
                            prd->attenuation.x *= prd->ss_alpha.x / min_alpha;
                            prd->ss_alpha.x = min_alpha;
                        }
                        if(prd->ss_alpha.y<min_alpha)
                        {
                            prd->attenuation.y *= prd->ss_alpha.y / min_alpha;
                            prd->ss_alpha.y = min_alpha;
                        }
                        if(prd->ss_alpha.z<min_alpha)
                        {
                            prd->attenuation.z *= prd->ss_alpha.z / min_alpha;
                            prd->ss_alpha.z = min_alpha;
                        }
                        //prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation/prd->sssAttenBegin) * prd->ss_alpha, prd->sigma_t, prd->channelPDF);
                        prd->maxDistance = DisneyBSDF::sample_scatter_distance(prd->attenuation/prd->sssAttenBegin,prd->sigma_t*prd->ss_alpha, prd->sigma_t,prd->seed,prd->channelPDF);
                        going_in_to_sss = true;
                        //here is the place caused inf ray:fixed
//                        auto min_sg = fmax(fmin(fmin(prd->sigma_t.x, prd->sigma_t.y), prd->sigma_t.z), 1e-8f);
//
//                        //what should be the right value???
//                        prd->maxDistance = max(prd->maxDistance, 10/min_sg);
                        //printf("maxdist:%f\n",prd->maxDistance);
                        // already calculated in BxDF
                        prd->pushMat(prd->sigma_t, prd->ss_alpha);
                        prd->isSS = true;
                        prd->scatterDistance = mats.scatterDistance;
                    }


                    prd->scatterStep = mats.scatterStep;
            }
            else{

                outToIn = false;
                inToOut = true;
                float3 trans = make_float3(1,1,1);
                vec3 sigma_t, ss_alpha;
                prd->readMat(sigma_t, ss_alpha);

                if(prd->curMatIdx==0)
                { 
                    trans = vec3(1.0f); 
                }
                else if (ss_alpha.x<0.0f) { // Glass
                    trans = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                } else {
                    trans = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);

                }
//                printf("%f,%f,%f\n",trans.x, trans.y, trans.z);
                prd->attenuation *= trans;
                CUR_TOTAL_TRANS  *= trans;
//                if(prd->print_info)
//    {
//        printf("hit and going out,depth:%d; Attenuation : %f,%f,%f; \n ", prd->depth,prd->attenuation.x, prd->attenuation.y, prd->attenuation.z);
//    }

                if(prd->curMatIdx>0)
                {
                    prd->popMat(sigma_t, ss_alpha);

                    prd->medium = (prd->curMatIdx == 0) ? DisneyBSDF::PhaseFunctions::vacuum
                                                        : DisneyBSDF::PhaseFunctions::isotropic;
                    if(prd->curMatIdx==0)
                    {
                        prd->isSS = false;
                        prd->maxDistance = 1e16;
                    }
                    else if (ss_alpha.x < 0.0f) {
                        prd->isSS = false;
                        prd->maxDistance = 1e16;
                    } else //next ray in 3s object
                    {
                        prd->isSS = true;
                        //prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation/prd->sssAttenBegin) * ss_alpha,
                                                                       //sigma_t, prd->channelPDF);
                        prd->maxDistance = DisneyBSDF::sample_scatter_distance(prd->attenuation/prd->sssAttenBegin,
                                                                               sigma_t*ss_alpha, sigma_t,prd->seed,prd->channelPDF);
                    }
                }else
                {
                    prd->medium = DisneyBSDF::PhaseFunctions::vacuum;
                    prd->isSS = false;
                    prd->maxDistance = 1e16;
                }
            }
        }else{
            if(prd->medium == DisneyBSDF::PhaseFunctions::isotropic){
                    vec3 trans = vec3(1.0f);
                    vec3 sigma_t, ss_alpha;
                    prd->readMat(sigma_t, ss_alpha);
                    prd->isSS = false;
                    if(prd->curMatIdx==0)
                    {
                        prd->maxDistance = 1e16f;
                    }
                    else if (ss_alpha.x<0.0f) { // Glass
                        trans = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                        vec3 channelPDF = vec3(1.0f/3.0f);
                        prd->maxDistance = mats.scatterStep>0.5f? DisneyBSDF::SampleDistance2(prd->seed, sigma_t, sigma_t, channelPDF) : 1e16f;
                    } else { // SSS
                        trans = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
                        //prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation/prd->sssAttenBegin) * ss_alpha, sigma_t, prd->channelPDF);
                        prd->maxDistance = DisneyBSDF::sample_scatter_distance(prd->attenuation/prd->sssAttenBegin,sigma_t*ss_alpha, sigma_t,prd->seed,prd->channelPDF);
                        prd->isSS = true;
                    }
                    prd->attenuation *= trans;
                    CUR_TOTAL_TRANS  *= trans;
            }
            else
            {
                prd->isSS = false;
                prd->medium = DisneyBSDF::PhaseFunctions::vacuum;
                prd->channelPDF = vec3(1.0f/3.0f);
                prd->maxDistance = 1e16f;
            }
        }
    }

    prd->medium = next_ray_is_going_inside?DisneyBSDF::PhaseFunctions::isotropic : (prd->curMatIdx==0?DisneyBSDF::PhaseFunctions::vacuum : DisneyBSDF::PhaseFunctions::isotropic);
    prd->countEmitted = false;
    prd->attenuation *= reflectance;
    if(mats.subsurface>0 && (mats.thin>0.5 || mats.doubleSide>0.5) && istransmission){
        CUR_TOTAL_TRANS *= reflectance;
    }
    prd->depth++;

//    if(prd->depth_diff>=3)
//        mats.roughness = clamp(mats.roughness, 0.5f,0.99f);

    auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {

        const auto& L = _wi_; // pre-normalized
        const vec3& V = _wo_; // pre-normalized
        auto& rd = reinterpret_cast<vec3&>(prd->aov[0]);
        auto& rs = reinterpret_cast<vec3&>(prd->aov[1]);
        auto& rt = reinterpret_cast<vec3&>(prd->aov[2]);

        float3 lbrdf = DisneyBSDF::EvaluateDisney3(vec3(1.0f), mats, L, V, T, B, N,prd->geometryNormal,
            mats.thin > 0.5f, flag == DisneyBSDF::transmissionEvent ? inToOut : next_ray_is_going_inside, thisPDF, rrPdf,
            dot(N, L), rd, rs, rt);

        return lbrdf;
    };

    vec3 auxRadiance = {};
    auto taskAux = [&](const vec3& radiance) {
        auxRadiance = auxRadiance + radiance;
    };

    ShadowPRD shadowPRD {};
    shadowPRD.seed = prd->seed ^ 0x9e3779b9u;
    shadowPRD.depth = prd->depth;
    shadowPRD.attanuation = make_float3(1.0f, 1.0f, 1.0f);
    shadowPRD.nonThinTransHit = (mats.thin < 0.5f && mats.specTrans > 0) ? 1 : 0;

    float3 frontPos, backPos;
    float3 sfrontPos, sbackPos;
    if (wldOffset > 0) {
        SelfIntersectionAvoidance::offsetSpawnPoint( frontPos, backPos, wldPos, prd->geometryNormal, wldOffset);
    } else {
        frontPos = wldPos;
        backPos = wldPos;
    }

    shadowPRD.origin = dot(wi, vec3(prd->geometryNormal)) > 0 ? frontPos : backPos;
    shadowPRD.origin = (isSS&&istransmission&&mats.thin<0.5&&mats.subsurface>0)?frontPos : shadowPRD.origin;
    shadowPRD.origin = shadowPRD.origin + float3(bezierOff);
    auto shadingP = frontPos + params.cam.eye; // world space
    if(mats.subsurface>0 && (mats.thin>0.5 || mats.doubleSide>0.5) && istransmission){
        shadingP = backPos + params.cam.eye;
    }

    prd->radiance = {};
    prd->direction = normalize(wi);
    prd->origin = dot(prd->direction, prd->geometryNormal) > 0 ? frontPos : backPos;

    prd->_tmin_ = 0.0f;


    float3 radianceNoShadow = {};
    float3* dummy_prt = nullptr;
    if (mats.shadowReceiver > 0.5f) {
        dummy_prt = &radianceNoShadow;
    }

    prd->lightmask = DefaultMatMask;

    shadowPRD.ShadowNormal = dot(wi, vec3(prd->geometryNormal)) > 0 ? prd->geometryNormal:-prd->geometryNormal;
    if(prd->hit_type==DIFFUSE_HIT && prd->diffDepth <=1 ) {
        uint8_t diffuse_sample_count = 1;
        for (auto i=0; i<diffuse_sample_count; ++i) {
            //shadowPRD.radiance += (coming_out_from_sss==true && mats.thin<0.5)? float3(mats.basecolor * mats.subsurface) * 0.01f:make_float3(0,0,0);
            mats.subsurface = coming_out_from_sss?0:mats.subsurface;
            mats.specular = coming_out_from_sss?0:mats.specular;
            DirectLighting<true>(shadowPRD, shadingP, coming_out_from_sss?prd->sssDirBegin:ray_dir, evalBxDF, &taskAux, dummy_prt);

        }
        float3 weight = CUR_TOTAL_TRANS * 1.0f / diffuse_sample_count * (going_in_to_sss?0:1);
        prd->radiance = shadowPRD.radiance * weight;
        if (prd->__aov__) {
            auxRadiance = auxRadiance * vec3(weight);
            prd->aov[0] *= auxRadiance;
            prd->aov[1] *= auxRadiance;
            prd->aov[2] *= auxRadiance;
        }
    }
    else {
        //shadowPRD.radiance += (coming_out_from_sss==true && mats.thin<0.5)? float3(mats.basecolor * mats.subsurface) * 0.05f:make_float3(0,0,0);
        mats.subsurface = coming_out_from_sss?0:mats.subsurface;
        mats.specular = coming_out_from_sss?0:mats.specular;
        DirectLighting<true>(shadowPRD, shadingP, coming_out_from_sss?prd->sssDirBegin:ray_dir, evalBxDF, &taskAux, dummy_prt);

        float3 weight = CUR_TOTAL_TRANS * (going_in_to_sss?0:1);
        prd->radiance = shadowPRD.radiance * weight;
        if (prd->__aov__) {
            prd->aov[0] *= auxRadiance * vec3(weight);
            prd->aov[1] *= auxRadiance * vec3(weight);
            prd->aov[2] *= auxRadiance * vec3(weight);
        }
    }
    if(mats.shadowReceiver > 0.5f)
    {
      auto radiance = length(prd->radiance);
      prd->radiance.x = radiance;//the light contribution received with shadow attenuation
      prd->radiance.y = length(radianceNoShadow);
      prd->radiance.z = 0;
      prd->done = true;
    }
//    if(prd->print_info)
//    {
//        printf("radiance at hit point,depth:%d; radiance : %f,%f,%f; \n ", prd->depth-1,prd->radiance.x, prd->radiance.y, prd->radiance.z);
//    }
//    prd->direction = normalize(wi);
//
//    prd->origin = dot(prd->direction, prd->geometryNormal) > 0.0f ? frontPos : backPos;

    if (prd->medium != DisneyBSDF::vacuum) {
        prd->_mask_ = (uint8_t)(EverythingMask ^ VolumeMatMask);
    } else {
        prd->_mask_ = EverythingMask;
    }

    prd->radiance += CUR_TOTAL_TRANS  * mats.emission;
    if(lengthSquared(mats.emission)>0)
    {
      prd->done = true;
    }
}
