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

static __inline__ __device__ bool isBadVector(const vec3& vector) {

    for (size_t i=0; i<3; ++i) {
        if(!isfinite(vector[i])) {
            return true;
        }
    }
    return dot(vector, vector) <= 0.0f;
}

static __inline__ __device__ bool isBadVector(const float3& vector) {
    return isBadVector(reinterpret_cast<const vec3&>(vector));
}

extern "C" __global__ void __anyhit__shadow_cutout()
{

    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint               primIdx = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    ShadowPRD* prd = getPRD<ShadowPRD>();
    MatInput attrs{};

    bool sphere_external_ray = false;

#if (_SPHERE_)

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, primIdx, sbtGASIndex, 0.f, &q );

    float3 _pos_world_      = P;
    float3 _pos_object_     = optixTransformPointFromWorldToObjectSpace( _pos_world_ );

    float3& _center_object_ = *(float3*)&q; 

    float3 _normal_object_  = ( _pos_object_ - _center_object_ ) / q.w;
    float3 _normal_world_   = normalize( optixTransformNormalFromObjectToWorldSpace( _normal_object_ ) );

    auto _origin_object_ = optixGetObjectRayOrigin();
    sphere_external_ray = length(_origin_object_ - _center_object_) > q.w;

    float3 N = _normal_world_;
    N = faceforward( N, -ray_dir, N );

    attrs.pos = P;
    attrs.nrm = N;
    attrs.uv = sphereUV(_normal_object_, false);

    attrs.clr = {};
    attrs.tang = {};
    attrs.instPos = {}; //rt_data->instPos[inst_idx2];
    attrs.instNrm = {}; //rt_data->instNrm[inst_idx2];
    attrs.instUv = {}; //rt_data->instUv[inst_idx2];
    attrs.instClr = {}; //rt_data->instClr[inst_idx2];
    attrs.instTang = {}; //rt_data->instTang[inst_idx2];

    unsigned short isLight = 0;
#else
    size_t inst_idx = optixGetInstanceIndex();
    size_t vert_aux_offset = rt_data->auxOffset[inst_idx];
    size_t vert_idx_offset = vert_aux_offset + primIdx*3;

    float3 _vertices_[3];
    optixGetTriangleVertexData( gas, primIdx, sbtGASIndex, 0, _vertices_);

    const float3& v0 = _vertices_[0];
    const float3& v1 = _vertices_[1];
    const float3& v2 = _vertices_[2];

    float3 N_Local = normalize( cross( normalize(v1-v0), normalize(v2-v1) ) );
    
    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();

    float3 n0 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+0 ]) );
    n0 = dot(n0, N_Local)>0.8f?n0:N_Local;
    float3 n1 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+1 ]) );
    n1 = dot(n1, N_Local)>0.8f?n1:N_Local;
    float3 n2 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+2 ]) );
    n2 = dot(n2, N_Local)>0.8f?n2:N_Local;

    N_Local = normalize(interp(barys, n0, n1, n2));
    float3 N_World = optixTransformNormalFromObjectToWorldSpace(N_Local);

    if (isBadVector(N_World)) 
    {  
        N_World = DisneyBSDF::SampleScatterDirection(prd->seed);
    }

    float3 N = faceforward( N_World, -ray_dir, N_World );
    
    attrs.pos = P;
    attrs.nrm = N;

    const float3& uv0  = decodeColor( rt_data->uv[ vert_idx_offset+0 ]   );
    const float3& uv1  = decodeColor( rt_data->uv[ vert_idx_offset+1 ]   );
    const float3& uv2  = decodeColor( rt_data->uv[ vert_idx_offset+2 ]   );
    const float3& clr0 = decodeColor( rt_data->clr[ vert_idx_offset+0 ]  );
    const float3& clr1 = decodeColor( rt_data->clr[ vert_idx_offset+1 ]  );
    const float3& clr2 = decodeColor( rt_data->clr[ vert_idx_offset+2 ]  );
    const float3& tan0 = decodeNormal( rt_data->tan[ vert_idx_offset+0 ] );
    const float3& tan1 = decodeNormal( rt_data->tan[ vert_idx_offset+1 ] );
    const float3& tan2 = decodeNormal( rt_data->tan[ vert_idx_offset+2 ] );

    attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = interp(barys, tan0, tan1, tan2);
    attrs.tang = optixTransformVectorFromObjectToWorldSpace(attrs.tang);
    attrs.rayLength = optixGetRayTmax();

    attrs.instPos =  decodeColor( rt_data->instPos[inst_idx] );
    attrs.instNrm =  decodeColor( rt_data->instNrm[inst_idx] );
    attrs.instUv =   decodeColor( rt_data->instUv[inst_idx]  );
    attrs.instClr =  decodeColor( rt_data->instClr[inst_idx] );
    attrs.instTang = decodeColor( rt_data->instTang[inst_idx]);

    unsigned short isLight = 0;//rt_data->lightMark[vert_aux_offset + primIdx];
#endif

    attrs.pos = attrs.pos + vec3(params.cam.eye);
    //MatOutput mats = evalMaterial(rt_data->textures, rt_data->uniforms, attrs);
    MatOutput mats = optixDirectCall<MatOutput, cudaTextureObject_t[], float4*, const MatInput&>( rt_data->dc_index, rt_data->textures, rt_data->uniforms, attrs );

    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
    }

    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01f,0.99f);

    /* MODME */
    auto basecolor = mats.basecolor;
    auto metallic = mats.metallic;
    auto roughness = mats.roughness;
    auto subsurface = mats.subsurface;
    auto specular = mats.specular;
    auto specularTint = mats.specularTint;
    auto anisotropic = mats.anisotropic;
    auto sheen = mats.sheen;
    auto sheenTint = mats.sheenTint;
    auto clearcoat = mats.clearcoat;
    auto opacity = mats.opacity;
    auto flatness = mats.flatness;
    auto specTrans = mats.specTrans;
    auto scatterDistance = mats.scatterDistance;
    auto ior = mats.ior;
    auto thin = mats.thin;
    auto doubleSide = mats.doubleSide;
    auto sssParam = mats.sssParam;
    auto scatterStep = mats.scatterStep;


    if(params.simpleRender==true)
        opacity = 0;
    //opacity = clamp(opacity, 0.0f, 0.99f);
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99f || isLight == 1) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
        return;
    }
    else
    {
        //roll a dice
        float p = rnd(prd->seed);

        float skip = opacity;
        #if (_SPHERE_)
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

                float nDi = fabs(dot(N,normalize(ray_dir)));
                vec3 fakeTrans = vec3(1)-BRDFBasics::fresnelSchlick(vec3(1) - mats.transColor,nDi);
                prd->attanuation = prd->attanuation * fakeTrans;

                #if (_SPHERE_)
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

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();

    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint               primIdx = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    MatInput attrs{};
    float estimation = 0;
#if (_SPHERE_)

    unsigned short isLight = 0;

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, primIdx, sbtGASIndex, 0.0f, &q );

    float3& sphere_center = *(float3*)&q;

    float3 objPos   = optixTransformPointFromWorldToObjectSpace(P);
    float3 objNorm  = normalize( ( objPos - sphere_center ) / q.w );

    objPos = sphere_center + objNorm * q.w;

    const float c0 = 5.9604644775390625E-8f;
    const float c1 = 1.788139769587360206060111522674560546875E-7f;
    const float c2 = 1.19209317972490680404007434844970703125E-7f;

    auto fma = [](auto a, auto b, auto c) -> auto {
        return a * b + c;
    };

    vec3 objErr = fma( vec3( c0 ), abs( sphere_center ), vec3( c1 * q.w * 2.0f ) );
    float objOffset = dot( objErr, abs( objNorm ) );

    float3 wldPos, wldNorm; float wldOffset;
    SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );

    P = wldPos;
    float3 N = wldNorm;

    prd->geometryNormal = N;

    attrs.pos = P;
    attrs.nrm = N;
    attrs.uv = sphereUV(objNorm, false);

    attrs.clr = {};
    attrs.tang = {};
    attrs.instPos = {}; //rt_data->instPos[inst_idx2];
    attrs.instNrm = {}; //rt_data->instNrm[inst_idx2];
    attrs.instUv = {}; //rt_data->instUv[inst_idx2];
    attrs.instClr = {}; //rt_data->instClr[inst_idx2];
    attrs.instTang = {}; //rt_data->instTang[inst_idx2];

#else

    size_t inst_idx = optixGetInstanceIndex();
    size_t vert_aux_offset = rt_data->auxOffset[inst_idx];
    size_t vert_idx_offset = vert_aux_offset + primIdx*3;
    //size_t tri_aux_offset = rt_data->auxTriOffset[inst_idx];
    //size_t tri_idx_offset = tri_aux_offset + primIdx;
    //size_t vidx0 = rt_data->vidx[tri_idx_offset * 3 + 0];
    //size_t vidx1 = rt_data->vidx[tri_idx_offset * 3 + 1];
    //size_t vidx2 = rt_data->vidx[tri_idx_offset * 3 + 2];

    unsigned short isLight = 0;//rt_data->lightMark[vert_aux_offset + primIdx];

    float3 _vertices_[3];
    optixGetTriangleVertexData( gas, primIdx, sbtGASIndex, 0, _vertices_);
    
    const float3& v0 = _vertices_[0];
    const float3& v1 = _vertices_[1];
    const float3& v2 = _vertices_[2];

    float3 objPos, objNorm; float objOffset; 
    //SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset );
    float2 barys = optixGetTriangleBarycentrics();
    SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, v0, v1, v2, barys );

    float3 wldPos, wldNorm; float wldOffset;
    SelfIntersectionAvoidance::transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset );

    /* MODMA */
    P = wldPos;
    attrs.pos = P;

    const float3& N_Local = objNorm;
    float3 N = wldNorm;

    if (isBadVector(N)) 
    {  
        N = normalize(DisneyBSDF::SampleScatterDirection(prd->seed));
        N = faceforward( N, -ray_dir, N );
    }
    prd->geometryNormal = N;

    attrs.nrm = N;

    const float3& uv0  = decodeColor( rt_data->uv[ vert_idx_offset+0 ] );
    const float3& uv1  = decodeColor( rt_data->uv[ vert_idx_offset+1 ] );
    const float3& uv2  = decodeColor( rt_data->uv[ vert_idx_offset+2 ] );
    const float3& clr0 = decodeColor( rt_data->clr[ vert_idx_offset+0 ] );
    const float3& clr1 = decodeColor( rt_data->clr[ vert_idx_offset+1 ] );
    const float3& clr2 = decodeColor( rt_data->clr[ vert_idx_offset+2 ] );
    const float3& tan0 = decodeNormal( rt_data->tan[ vert_idx_offset+0 ] );
    const float3& tan1 = decodeNormal( rt_data->tan[ vert_idx_offset+1 ] );
    const float3& tan2 = decodeNormal( rt_data->tan[ vert_idx_offset+2 ] );
    float tri_area = length(cross(_vertices_[1]-_vertices_[0], _vertices_[2]-_vertices_[1]));
    float uv_area = length(cross(uv1 - uv0, uv2-uv0));
    estimation = uv_area * 4096.0f*4096.0f / (tri_area + 1e-6);
        attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = normalize(interp(barys, tan0, tan1, tan2));
    attrs.tang = optixTransformVectorFromObjectToWorldSpace(attrs.tang);

    attrs.instPos =  decodeColor( rt_data->instPos[inst_idx] );
    attrs.instNrm =  decodeColor( rt_data->instNrm[inst_idx] );
    attrs.instUv =   decodeColor( rt_data->instUv[inst_idx]  );
    attrs.instClr =  decodeColor( rt_data->instClr[inst_idx] );
    attrs.instTang = decodeColor( rt_data->instTang[inst_idx]);
    attrs.rayLength = optixGetRayTmax();

    float3 n0 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+0 ]) );
    n0 = n0;

    float3 n1 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+1 ]) );
    n1 = n1;

    float3 n2 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+2 ]) );
    n2 = n2;

    auto N_smooth = normalize(interp(barys, n0, n1, n2));
    attrs.N = optixTransformNormalFromObjectToWorldSpace(N_smooth);


#endif

    attrs.pos = attrs.pos + vec3(params.cam.eye);
    if(! (length(attrs.tang)>0.0f) )
    {
      Onb a(attrs.N);
      attrs.T = a.m_tangent;
    }
    else
    {
      attrs.T = attrs.tang;
    }
    attrs.V = -(ray_dir);
    //MatOutput mats = evalMaterial(rt_data->textures, rt_data->uniforms, attrs);
    MatOutput mats = optixDirectCall<MatOutput, cudaTextureObject_t[], float4*, const MatInput&>( rt_data->dc_index, rt_data->textures, rt_data->uniforms, attrs );

    if (prd->test_distance) {
    
        if(mats.opacity>0.99f) { // it's actually transparency not opacity
            prd->_tmin_ = optixGetRayTmax();
        } else if(rnd(prd->seed)<mats.opacity) {
            prd->_tmin_ = optixGetRayTmax();
        } else {
            prd->test_distance = false;
            prd->maxDistance = optixGetRayTmax();
        }
        return;
    }

#if _SPHERE_

    if(mats.doubleSide>0.5f||mats.thin>0.5f){
        N = faceforward( N, -ray_dir, N );
        prd->geometryNormal = N;
    }

#else
    n0 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+0 ]) );
    n0 = dot(n0, N_Local)>(1-mats.smoothness)?n0:N_Local;

    n1 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+1 ]) );
    n1 = dot(n1, N_Local)>(1-mats.smoothness)?n1:N_Local;

    n2 = normalize( decodeNormal(rt_data->nrm[ vert_idx_offset+2 ]) );
    n2 = dot(n2, N_Local)>(1-mats.smoothness)?n2:N_Local;

    N_smooth = normalize(interp(barys, n0, n1, n2));
    N = optixTransformNormalFromObjectToWorldSpace(N_smooth);

    if(mats.doubleSide>0.5f||mats.thin>0.5f){
        N = faceforward( N, -ray_dir, N );
        prd->geometryNormal = faceforward( prd->geometryNormal, -ray_dir, prd->geometryNormal );
    }
#endif

    attrs.nrm = N;
    float term = log2(optixGetRayTmax()*prd->pixel_area*sqrt(estimation))/4.0f;
//    printf("rayDist:%f, tex_per_area:%f, term:%f, pixel_area:%f\n", optixGetRayTmax(),
//           sqrt(estimation), term, prd->pixel_area);
    //mats.nrm = normalize(mix(mats.nrm, vec3(0,0,1), clamp(term,0.0f,1.0f)));
    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01f,0.99f);
    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
    }
//    if(dot(vec3(ray_dir), vec3(N)) * dot(vec3(ray_dir), vec3(prd->geometryNormal))<0)
//    {
//      N = prd->geometryNormal;
//    }

    if (prd->trace_denoise_albedo) {

        if(0.0f == mats.roughness) {
            prd->tmp_albedo = make_float3(1.0f);
        } else {
            prd->tmp_albedo = mats.basecolor;
        }
    }

    if (prd->trace_denoise_normal) {
        prd->tmp_normal = N;
    }

    bool next_ray_is_going_inside = false;
    mats.sssParam = mats.subsurface>0 ? mats.subsurface*mats.sssParam : mats.sssParam;
    mats.subsurface = mats.subsurface>0 ? 1 : 0;

    /* MODME */

    if(prd->diffDepth>=1)
        mats.roughness = clamp(mats.roughness, 0.2,0.99);
    if(prd->diffDepth>=2)
        mats.roughness = clamp(mats.roughness, 0.3,0.99);
    if(prd->diffDepth>=3)
        mats.roughness = clamp(mats.roughness, 0.5,0.99);

    
    if(prd->isSS == true) {
        mats.basecolor = vec3(1.0f);
        mats.roughness = 1.0;
        mats.anisotropic = 0;
        mats.sheen = 0;
        mats.clearcoat = 0;
        mats.specTrans = 0;
        mats.ior = 1;
    }
    if(prd->isSS == true && mats.subsurface>0 && dot(-normalize(ray_dir), N)>0)
    {
       prd->attenuation2 = make_float3(0,0,0);
       prd->attenuation = make_float3(0,0,0);
       prd->radiance = make_float3(0,0,0);
       prd->done = true;
       return;
    }
    if(prd->isSS == true  && mats.subsurface==0 )
    {
        prd->passed = true;
        prd->samplePdf = 1.0f;
        prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
        prd->readMat(prd->sigma_t, prd->ss_alpha);
        auto trans = DisneyBSDF::Transmission2(prd->sigma_s(), prd->sigma_t, prd->channelPDF, optixGetRayTmax(), true);
        prd->attenuation2 *= trans;
        prd->attenuation *= trans;
        //prd->origin = P;
        prd->direction = ray_dir;
        //auto n = prd->geometryNormal;
        //n = faceforward(n, -ray_dir, n);
        prd->_tmin_ = optixGetRayTmax();
        return;
    }

    prd->attenuation2 = prd->attenuation;
    prd->countEmitted = false;
    prd->prob2 = prd->prob;
    prd->passed = false;

    if(mats.opacity>0.99f)
    {

        if (prd->curMatIdx > 0) {
          vec3 sigma_t, ss_alpha;
          //vec3 sigma_t, ss_alpha;
          prd->readMat(sigma_t, ss_alpha);
          if (ss_alpha.x < 0.0f) { // is inside Glass
            prd->attenuation *= DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
          } else {
            prd->attenuation *= DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
          }
        }
        prd->attenuation2 = prd->attenuation;
        prd->passed = true;
        prd->adepth++;
        //prd->samplePdf = 0.0f;
        prd->radiance = make_float3(0.0f);
        prd->alphaHit = true;
        prd->_tmin_ = optixGetRayTmax();
        return;
    }
    if(mats.opacity<=0.99f)
    {
      //we have some simple transparent thing
      //roll a dice to see if just pass
      if(rnd(prd->seed)<mats.opacity)
      {
        if (prd->curMatIdx > 0) {
          vec3 sigma_t, ss_alpha;
          //vec3 sigma_t, ss_alpha;
          prd->readMat(sigma_t, ss_alpha);
          if (ss_alpha.x < 0.0f) { // is inside Glass
            prd->attenuation *= DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
          } else {
            prd->attenuation *= DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
          }
        }
        prd->attenuation2 = prd->attenuation;
        prd->passed = true;
        prd->adepth++;
        //prd->samplePdf = 0.0f;
        //you shall pass!
        prd->radiance = make_float3(0.0f);
        prd->_tmin_ = optixGetRayTmax();
        prd->alphaHit = true;

        prd->prob *= 1;
        prd->countEmitted = false;
        return;
      }
    }
    if(prd->depth==0&&mats.flatness>0.5)
    {
        prd->radiance = make_float3(0.0f);
        prd->done = true;
        return;
    }
    
    float is_refl;
    float3 inDir = ray_dir;
    vec3 wi = vec3(0.0f);
    float pdf = 0.0f;
    float rPdf = 0.0f;
    float fPdf = 0.0f;
    float rrPdf = 0.0f;

    float3 T = attrs.tang;
    float3 B;
    if(length(T)>0)
    {
        B = cross(N, T);
    } else
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

    //sssColor = mix(basecolor, sssColor, subsurface);

    while(DisneyBSDF::SampleDisney2(
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
            reflectance = fPdf>0?reflectance/fPdf:vec3(0.0f);
            prd->done = fPdf>0?true:prd->done;
            flag = DisneyBSDF::scatterEvent;
        }
        
    prd->samplePdf = fPdf;
    reflectance = fPdf>0?reflectance/fPdf:vec3(0.0f);
    prd->done = fPdf>0?prd->done:true;
    prd->isSS = isSS;
    pdf = 1.0;
    if(isDiff || prd->diffDepth>0){
        prd->diffDepth++;
    }


    prd->passed = false;
    bool inToOut = false;
    bool outToIn = false;

    bool istransmission = dot(vec3(prd->geometryNormal), vec3(wi)) * dot(vec3(prd->geometryNormal), vec3(-normalize(ray_dir)))<0;
    //istransmission = (istransmission && thin<0.5 && mats.doubleSide==false);
    if(istransmission || flag == DisneyBSDF::diracEvent) {
    //if(flag == DisneyBSDF::transmissionEvent || flag == DisneyBSDF::diracEvent) {
        next_ray_is_going_inside = dot(vec3(prd->geometryNormal),vec3(wi))<=0;
    }
    prd->max_depth = ((prd->depth==0 && isSS) || (prd->depth>0 && (mats.specTrans>0||mats.isHair>0)) )?16:prd->max_depth;
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
            prd->attenuation2 *= trans;
        }

        next_ray_is_going_inside = false;

    }else{
    
        //if(flag == DisneyBSDF::transmissionEvent || flag == DisneyBSDF::diracEvent) {
        if(istransmission || flag == DisneyBSDF::diracEvent) {
            if(next_ray_is_going_inside){

                    outToIn = true;
                    inToOut = false;

                    prd->medium = DisneyBSDF::PhaseFunctions::isotropic;

                    if (prd->curMatIdx > 0) {
                        vec3 sigma_t, ss_alpha;
                        //vec3 sigma_t, ss_alpha;0
                        prd->readMat(sigma_t, ss_alpha);
                        if (ss_alpha.x < 0.0f) { // is inside Glass
                            prd->attenuation *= DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                        } else {
                            prd->attenuation *= DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
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

                        prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * prd->ss_alpha, prd->sigma_t, prd->channelPDF);
                        //here is the place caused inf ray:fixed
                        auto min_sg = fmax(fmin(fmin(prd->sigma_t.x, prd->sigma_t.y), prd->sigma_t.z), 1e-8f);
                        //what should be the right value???
                        //prd->maxDistance = max(prd->maxDistance, 10/min_sg);
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

                float3 trans;
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

                prd->attenuation *= trans;
                prd->attenuation2 *= trans;
                
                prd->popMat(sigma_t, ss_alpha);

                prd->medium = (prd->curMatIdx==0)? DisneyBSDF::PhaseFunctions::vacuum : DisneyBSDF::PhaseFunctions::isotropic;

                if(ss_alpha.x < 0.0f) 
                {
                    prd->isSS = false;
                    prd->maxDistance = 1e16;
                }
                else //next ray in 3s object
                {
                    prd->isSS = true;
                    prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * ss_alpha, sigma_t, prd->channelPDF);
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
                        prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * ss_alpha, sigma_t, prd->channelPDF);
                        prd->isSS = true;
                    }

                    prd->attenuation2 *= trans;
                    prd->attenuation *= trans;
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

    prd->medium = next_ray_is_going_inside?DisneyBSDF::PhaseFunctions::isotropic : prd->curMatIdx==0?DisneyBSDF::PhaseFunctions::vacuum : DisneyBSDF::PhaseFunctions::isotropic;
 

//    if(mats.thin>0.5f){
//        vec3 H = normalize(vec3(normalize(wi)) + vec3(-normalize(ray_dir)));
//        attrs.N = N;
//        attrs.T = cross(B,N);
//        attrs.L = vec3(normalize(wi));
//        attrs.V = vec3(-normalize(ray_dir));
//        attrs.H = normalize(H);
//        attrs.reflectance = reflectance;
//        attrs.fresnel = DisneyBSDF::DisneyFresnel(mats.basecolor, mats.metallic, mats.ior, mats.specularTint, dot(attrs.H, attrs.V), dot(attrs.H, attrs.L), false);
//        MatOutput mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
//        reflectance = mat2.reflectance;
//    }


    prd->countEmitted = false;
    prd->attenuation *= reflectance;
    if(mats.subsurface>0 && (mats.thin>0.5 || mats.doubleSide>0.5) && istransmission){
      prd->attenuation2 *= reflectance;
    }
    prd->depth++;

    if(prd->depth>=3)
        mats.roughness = clamp(mats.roughness, 0.5f,0.99f);

    auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {

        const auto& L = _wi_; // pre-normalized
        const vec3& V = _wo_; // pre-normalized
        vec3 rd, rs, rt; // captured by lambda

        float3 lbrdf = DisneyBSDF::EvaluateDisney2(vec3(1.0f), mats, L, V, T, B, N,prd->geometryNormal,
            mats.thin > 0.5f, flag == DisneyBSDF::transmissionEvent ? inToOut : next_ray_is_going_inside, thisPDF, rrPdf,
            dot(N, L), rd, rs, rt);

        prd->radiance_d = rd;
        prd->radiance_s = rs;
        prd->radiance_t = rt;
//        MatOutput mat2;
//        if(mats.thin>0.5f){
//            vec3 H = normalize(vec3(normalize(L)) + V);
//            attrs.N = N;
//            attrs.T = cross(B,N);
//            attrs.L = vec3(normalize(L));
//            attrs.V = V;
//            attrs.H = normalize(H);
//            attrs.reflectance = lbrdf;
//            attrs.fresnel = DisneyBSDF::DisneyFresnel( mats.basecolor, mats.metallic, mats.ior, mats.specularTint, dot(attrs.H, attrs.V), dot(attrs.H, attrs.L), false);
//            mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
//        }

        return lbrdf;

    };

    auto taskAux = [&](const vec3& radiance) {
        prd->radiance_d *= radiance;
        prd->radiance_s *= radiance;
        prd->radiance_t *= radiance;
    };

    ShadowPRD shadowPRD {};
    shadowPRD.seed = prd->seed;
    shadowPRD.attanuation = make_float3(1.0f, 1.0f, 1.0f);
    shadowPRD.nonThinTransHit = (mats.thin < 0.5f && mats.specTrans > 0) ? 1 : 0;

    float3 frontPos, backPos;
    SelfIntersectionAvoidance::offsetSpawnPoint( frontPos, backPos, wldPos, prd->geometryNormal, wldOffset );

    shadowPRD.origin = dot(-ray_dir, wldNorm) > 0 ? frontPos : backPos;
    //auto shadingP = rtgems::offset_ray(shadowPRD.origin + params.cam.eye,  prd->geometryNormal); // world space
    
    shadowPRD.origin = frontPos;
    if(mats.subsurface>0 && (mats.thin>0.5 || mats.doubleSide>0.5) && istransmission){
        shadowPRD.origin = backPos; //rtgems::offset_ray(P,  -prd->geometryNormal);
    }
    
    auto shadingP = rtgems::offset_ray(P + params.cam.eye,  prd->geometryNormal); // world space
    if(mats.subsurface>0 && (mats.thin>0.5 || mats.doubleSide>0.5) && istransmission){
        shadingP = rtgems::offset_ray(P + params.cam.eye,  -prd->geometryNormal);
    }

    prd->radiance = {};
    prd->direction = normalize(wi);
    prd->origin = dot(prd->direction, wldNorm) > 0 ? frontPos : backPos;


    float3 radianceNoShadow = {};
    float3* dummy_prt = nullptr;
    if (mats.shadowReceiver > 0.5f) {
        dummy_prt = &radianceNoShadow;
    }

    prd->lightmask = DefaultMatMask;
    DirectLighting<true>(prd, shadowPRD, shadingP, ray_dir, evalBxDF, &taskAux, dummy_prt);
    if(mats.shadowReceiver > 0.5f)
    {
      auto radiance = length(prd->radiance);
      prd->radiance.x = radiance;//the light contribution received with shadow attenuation
      prd->radiance.y = length(radianceNoShadow);
      prd->radiance.z = 0;
      prd->done = true;
    }

    prd->direction = normalize(wi);

    if(mats.thin<0.5f && mats.doubleSide<0.5f){
        //auto p_prim = vec3(prd->origin) + optixGetRayTmax() * vec3(prd->direction);
        //float3 p = p_prim;
        prd->origin = next_ray_is_going_inside? backPos : frontPos;
    }
    else {
        //auto p_prim = vec3(prd->origin) + optixGetRayTmax() * vec3(prd->direction);
        //float3 p = p_prim;
        prd->origin = dot(prd->direction, prd->geometryNormal) < 0? backPos : frontPos;
    }

    if (prd->medium != DisneyBSDF::vacuum) {
        prd->_mask_ = (uint8_t)(EverythingMask ^ VolumeMatMask);
    } else {
        prd->_mask_ = EverythingMask;
    }

    prd->radiance += mats.emission;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
