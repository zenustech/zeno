#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>

#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "DisneyBRDF.h"
#include "DisneyBSDF.h"

#include "IOMat.h"
#include "Shape.h"
#include "Lighting.h"


#define _SPHERE_ 0
#define _LIGHT_SOURCE_ 0

#define TRI_PER_MESH 4096
//COMMON_CODE

template<bool isDisplacement>
static __inline__ __device__ MatOutput evalMat(cudaTextureObject_t zenotex[], float4* uniforms, MatInput const &attrs) {

    /* MODMA */
    auto att_pos = attrs.pos;
    auto att_clr = attrs.clr;
    auto att_uv = attrs.uv;
    auto att_nrm = attrs.nrm;
    auto att_tang = attrs.tang;
    auto att_instPos = attrs.instPos;
    auto att_instNrm = attrs.instNrm;
    auto att_instUv = attrs.instUv;
    auto att_instClr = attrs.instClr;
    auto att_instTang = attrs.instTang;
    auto att_NoL      = attrs.NoL;
    auto att_LoV      = attrs.LoV;
    auto att_N        = attrs.N;
    auto att_T        = attrs.T;
    auto att_L        = attrs.L;
    auto att_V        = attrs.V;
    auto att_H        = attrs.H;
    auto att_reflectance = attrs.reflectance;
    auto att_fresnel  = attrs.fresnel;
    /** generated code here beg **/
    //GENERATED_BEGIN_MARK
    /* MODME */
    float mat_base = 1.0;
    vec3 mat_basecolor = vec3(1.0, 1.0, 1.0);
    float mat_metallic = 0.0;
    float mat_roughness = 0.5;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_anisoRotation = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearcoat = 0.0;
    float mat_clearcoatGloss = 0.0;
    float mat_clearcoatRoughness = 0.0;
    float mat_clearcoatIOR = 1.5;
    float mat_opacity = 0.0;
    float mat_specTrans = 0.0;
    float mat_ior = 1.0;
    float mat_scatterDistance = 0.0;
    float mat_flatness = 0.0;
    float mat_thin = 0.0;
    float mat_doubleSide= 0.0;
    float mat_scatterStep = 0.0f;
    float mat_smoothness = 0.0f;
    vec3  mat_sssColor = vec3(0.0f,0.0f,0.0f);
    vec3  mat_sssParam = vec3(0.0f,0.0f,0.0f);
    vec3  mat_normal = vec3(0.0f, 0.0f, 1.0f);
    float mat_emissionIntensity = float(0);
    vec3 mat_emission = vec3(1.0f, 1.0f, 1.0f);
    float mat_displacement = 0.0f;
    float mat_NoL = 1.0f;
    float mat_LoV = 1.0f;
    vec3 mat_reflectance = att_reflectance;
    //GENERATED_END_MARK
    /** generated code here end **/
    MatOutput mats;
    if constexpr(isDisplacement)
    {
        mats.reflectance = mat_reflectance;
        return mats;
    }else {
        /* MODME */
        mats.basecolor = mat_base * mat_basecolor;
        mats.metallic = clamp(mat_metallic, 0.0f, 1.0f);
        mats.roughness = clamp(mat_roughness, 0.01, 0.99);
        mats.subsurface = mat_subsurface;
        mats.specular = mat_specular;
        mats.specularTint = mat_specularTint;
        mats.anisotropic = clamp(mat_anisotropic, 0.0f, 1.0f);
        mats.anisoRotation = clamp(mat_anisoRotation, 0.0f, 1.0f);
        mats.sheen = mat_sheen;
        mats.sheenTint = mat_sheenTint;
        mats.clearcoat = clamp(mat_clearcoat, 0.0f, 1.0f);
        mats.clearcoatGloss = mat_clearcoatGloss;
        mats.clearcoatRoughness = clamp(mat_clearcoatRoughness, 0.01, 0.99);
        mats.clearcoatIOR = mat_clearcoatIOR;
        mats.opacity = mat_opacity;
        mats.nrm = mat_normal;
        mats.emission = mat_emissionIntensity * mat_emission;
        mats.specTrans = clamp(mat_specTrans, 0.0f, 1.0f);
        mats.ior = mat_ior;
        mats.scatterDistance = mat_scatterDistance;
        mats.flatness = mat_flatness;
        mats.thin = mat_thin;
        mats.doubleSide = mat_doubleSide;
        mats.sssColor = mat_sssColor;
        mats.sssParam = mat_sssParam;
        mats.scatterStep = mat_scatterStep;
        mats.smoothness = mat_smoothness;
        return mats;
    }
}

static __inline__ __device__ MatOutput evalMaterial(cudaTextureObject_t zenotex[], float4* uniforms, MatInput const &attrs)
{
    return evalMat<false>(zenotex, uniforms, attrs);
}

static __inline__ __device__ MatOutput evalGeometry(cudaTextureObject_t zenotex[], float4* uniforms, MatInput const &attrs)
{
    return evalMat<true>(zenotex, uniforms, attrs);
}

static __inline__ __device__ MatOutput evalReflectance(cudaTextureObject_t zenotex[], float4* uniforms, MatInput const &attrs)
{
    return evalMat<true>(zenotex, uniforms, attrs);
}
__forceinline__ __device__ float3 interp(float2 barys, float3 a, float3 b, float3 c)
{
    float w0 = 1 - barys.x - barys.y;
    float w1 = barys.x;
    float w2 = barys.y;
    return w0*a + w1*b + w2*c;
}

static __inline__ __device__ bool isBadVector(vec3& vector) {

    for (size_t i=0; i<3; ++i) {
        if(isnan(vector[i]) || isinf(vector[i])) {
            return true;
        }
    }
    return length(vector) <= 0;
}

static __inline__ __device__ bool isBadVector(float3& vector) {
    vec3 tmp = vector;
    return isBadVector(tmp);
}

static __inline__ __device__ float3 sphereUV(float3 &direction) {
    
    return float3 {
        atan2(direction.x, direction.z) / (2.0f*M_PIf) + 0.5f,
        direction.y * 0.5f + 0.5f, 0.0f
    };
} 

extern "C" __global__ void __anyhit__shadow_cutout()
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint              prim_idx = optixGetPrimitiveIndex();

    const float3 ray_orig        = optixGetWorldRayOrigin();
    const float3 ray_dir         = optixGetWorldRayDirection();

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const float3 P    = ray_orig + optixGetRayTmax() * ray_dir;
    const auto zenotex = rt_data->textures;

    RadiancePRD*  prd = getPRD();
    MatInput attrs{};

    if constexpr(_LIGHT_SOURCE_)
    {
        prd->attenuation2 = vec3(0.0f);
        prd->attenuation = vec3(0.0f);
        optixTerminateRay();
        return;
    }

#if (_SPHERE_)

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );

    float3 _pos_world_      = ray_orig + optixGetRayTmax() * ray_dir;
    float3 _pos_object_     = optixTransformPointFromWorldToObjectSpace( _pos_world_ );

    float3 _normal_object_  = ( _pos_object_ - make_float3( q ) ) / q.w;
    float3 _normal_world_   = normalize( optixTransformNormalFromObjectToWorldSpace( _normal_object_ ) );

    //float3 P = _pos_world_;
    float3 N = _normal_world_;
    N = faceforward( N, -ray_dir, N );

    attrs.pos = P;
    attrs.nrm = N;
    attrs.uv = sphereUV(_normal_object_);

    attrs.clr = {};
    attrs.tang = {};
    attrs.instPos = {}; //rt_data->instPos[inst_idx2];
    attrs.instNrm = {}; //rt_data->instNrm[inst_idx2];
    attrs.instUv = {}; //rt_data->instUv[inst_idx2];
    attrs.instClr = {}; //rt_data->instClr[inst_idx2];
    attrs.instTang = {}; //rt_data->instTang[inst_idx2];

    unsigned short isLight = 0;
#else

    int inst_idx2 = optixGetInstanceIndex();
    int inst_idx = rt_data->meshIdxs[inst_idx2];
    int vert_idx_offset = (inst_idx * TRI_PER_MESH + prim_idx)*3;

    float m16[16];
    m16[12]=0; m16[13]=0; m16[14]=0; m16[15]=1;
    optixGetObjectToWorldTransformMatrix(m16);
    mat4& meshMat = *reinterpret_cast<mat4*>(&m16);

    float3 _vertices_[3];
    optixGetTriangleVertexData( gas,
                                prim_idx,
                                sbtGASIndex,
                                0,
                                _vertices_);

    float3 av0 = _vertices_[0]; //make_float3(rt_data->vertices[vert_idx_offset + 0]);
    float3 av1 = _vertices_[1]; //make_float3(rt_data->vertices[vert_idx_offset + 1]);
    float3 av2 = _vertices_[2]; //make_float3(rt_data->vertices[vert_idx_offset + 2]);
    vec4 bv0 = vec4(av0.x, av0.y, av0.z, 1);
    vec4 bv1 = vec4(av1.x, av1.y, av1.z, 1);
    vec4 bv2 = vec4(av2.x, av2.y, av2.z, 1);
    bv0 = meshMat * bv0;
    bv1 = meshMat * bv1;
    bv2 = meshMat * bv2;
    float3 v0 = make_float3(bv0.x, bv0.y, bv0.z);
    float3 v1 = make_float3(bv1.x, bv1.y, bv1.z);
    float3 v2 = make_float3(bv2.x, bv2.y, bv2.z);

    float3 N_0 = normalize( cross( normalize(v1-v0), normalize(v2-v0) ) );

    if (isBadVector(N_0)) 
    {  
        //assert(false);
        N_0 = DisneyBSDF::SampleScatterDirection(prd->seed);
        N_0 = faceforward( N_0, -ray_dir, N_0 );
    }
    //float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    
    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();
    
    mat3 meshMat3x3(meshMat);
    float3 an0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
    vec3 bn0(an0);
    bn0 = meshMat3x3 * bn0;
    float3 n0 = make_float3(bn0.x, bn0.y, bn0.z);
    n0 = dot(n0, N_0)>0.8f?n0:N_0;

    float3 an1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
    vec3 bn1(an1);
    bn1 = meshMat3x3 * bn1;
    float3 n1 = make_float3(bn1.x, bn1.y, bn1.z);
    n1 = dot(n1, N_0)>0.8f?n1:N_0;

    float3 an2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));
    vec3 bn2(an2);
    bn2 = meshMat3x3 * bn2;
    float3 n2 = make_float3(bn2.x, bn2.y, bn2.z);
    n2 = dot(n2, N_0)>0.8f?n2:N_0;
    float3 uv0 = make_float3(rt_data->uv[ vert_idx_offset+0 ] );
    float3 uv1 = make_float3(rt_data->uv[ vert_idx_offset+1 ] );
    float3 uv2 = make_float3(rt_data->uv[ vert_idx_offset+2 ] );
    float3 clr0 = make_float3(rt_data->clr[ vert_idx_offset+0 ] );
    float3 clr1 = make_float3(rt_data->clr[ vert_idx_offset+1 ] );
    float3 clr2 = make_float3(rt_data->clr[ vert_idx_offset+2 ] );
    float3 atan0 = make_float3(rt_data->tan[ vert_idx_offset+0 ] );
    float3 atan1 = make_float3(rt_data->tan[ vert_idx_offset+1 ] );
    float3 atan2 = make_float3(rt_data->tan[ vert_idx_offset+2 ] );
    vec3 btan0(atan0);
    vec3 btan1(atan1);
    vec3 btan2(atan2);
    btan0 = meshMat3x3 * btan0;
    btan1 = meshMat3x3 * btan1;
    btan2 = meshMat3x3 * btan2;
    float3 tan0 = make_float3(btan0.x, btan0.y, btan0.z);
    float3 tan1 = make_float3(btan1.x, btan1.y, btan1.z);
    float3 tan2 = make_float3(btan2.x, btan2.y, btan2.z);
    
    N_0 = normalize(interp(barys, n0, n1, n2));
    float3 N = faceforward( N_0, -ray_dir, N_0 );

    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = N;
    attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = interp(barys, tan0, tan1, tan2);
    attrs.instPos = rt_data->instPos[inst_idx2];
    attrs.instNrm = rt_data->instNrm[inst_idx2];
    attrs.instUv = rt_data->instUv[inst_idx2];
    attrs.instClr = rt_data->instClr[inst_idx2];
    attrs.instTang = rt_data->instTang[inst_idx2];

    unsigned short isLight = rt_data->lightMark[inst_idx * TRI_PER_MESH + prim_idx];
#endif

    MatOutput mats = evalMaterial(zenotex, rt_data->uniforms, attrs);

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
    auto clearcoatGloss = mats.clearcoatGloss;
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
    }
    else
    {

        //roll a dice
        float p = rnd(prd->seed);
        if (p < opacity){
            optixIgnoreIntersection();
        }else{
            if(length(prd->shadowAttanuation) < 0.01f){
                prd->shadowAttanuation = vec3(0.0f);
                optixTerminateRay();
                return;
            }
            if(specTrans==0.0f){
                prd->shadowAttanuation = vec3(0.0f);
                optixTerminateRay();
                return;
            }
            //prd->shadowAttanuation = vec3(0,0,0);
            //optixTerminateRay();
            
            if(specTrans > 0.0f){
                if(thin == 0.0f && ior>1.0f)
                {
                    prd->nonThinTransHit++;
                }
                if(rnd(prd->seed)<(1-specTrans)||prd->nonThinTransHit>1)
                {
                    prd->shadowAttanuation = vec3(0,0,0);
                    optixTerminateRay();
                    return;
                }
                float nDi = fabs(dot(N,ray_dir));
                vec3 tmp = prd->shadowAttanuation;
                tmp = tmp * (vec3(1)-BRDFBasics::fresnelSchlick(vec3(1)-basecolor,nDi));
                prd->shadowAttanuation = tmp;

                optixIgnoreIntersection();
            }
        }

        prd->shadowAttanuation = vec3(0);
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
    if(prd->test_distance)
    {
        prd->vol_t1 = optixGetRayTmax();
        return;
    }
    prd->test_distance = false;

    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint              prim_idx = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    auto zenotex = rt_data->textures;

    if constexpr(_LIGHT_SOURCE_)
    {
        auto instanceId = optixGetInstanceId();
        auto isLightGAS = ( instanceId >= OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID-1 );

        const auto pType = optixGetPrimitiveType();
        //optixGetPrimitiveIndex();
        //assert(false);
        if (params.num_lights == 0 || !isLightGAS) {
            prd->depth += 1;
            prd->done = true;
            return;
        }

        uint light_idx = 0;

        if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {
            light_idx = prim_idx + params.firstSphereLightIdx;
        } else {
            auto rect_idx = prim_idx / 2;
            light_idx = rect_idx + params.firstRectLightIdx;
        }

        light_idx = min(light_idx, params.num_lights - 1);
        auto& light = params.lights[light_idx];

        float prevCDF = light_idx>0? params.lights[light_idx-1].CDF : 0.0f;
        float lightPickPDF = (light.CDF - prevCDF) / params.lights[params.num_lights-1].CDF;

        auto visible = (light.config & LightConfigVisible);

        if (!visible && prd->depth == 0) {
            auto pos = ray_orig + ray_dir * optixGetRayTmax();
            prd->geometryNormal = light.N;
            prd->offsetUpdateRay(pos, ray_dir); 
            return;
        }

        prd->depth += 1;
        prd->done = true;

        float3 lightDirection = optixGetWorldRayDirection(); //light_pos - P;
        float  lightDistance  = optixGetRayTmax();  //length(lightDirection);

        LightSampleRecord lsr;

        if (light.shape == 0) {
            light.rect.eval(&lsr, lightDirection, lightDistance, prd->origin);
        } else {
            light.sphere.eval(&lsr, lightDirection, lightDistance, prd->origin);
        }

        if (light.config & LightConfigDoubleside) {
            lsr.NoL = abs(lsr.NoL);
        }

        if (lsr.NoL > _FLT_EPL_) {

            if (1 == prd->depth) {
                if (light.config & LightConfigVisible) {
                    prd->radiance = light.emission;
                }
                prd->attenuation = vec3(1.0f); 
                prd->attenuation2 = vec3(1.0f);
                return;
            }

            float scatterPDF = prd->samplePdf; //BxDF direction PDF from previous hit
            float lightPDF = lightPickPDF * lsr.PDF;
            float misWeight = BRDFBasics::PowerHeuristic(scatterPDF, lightPDF);

            prd->radiance = light.emission * misWeight;
            // if (scatterPDF > __FLT_DENORM_MIN__) {
            //     prd->radiance /= scatterPDF;
            // }
        }
        return;
    }

    MatInput attrs{};

#if (_SPHERE_)

    unsigned short isLight = 0;

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, optixGetPrimitiveIndex(), sbtGASIndex, 0.0f, &q );

    float3 _pos_world_      = ray_orig + optixGetRayTmax() * ray_dir;
    float3 _pos_object_     = optixTransformPointFromWorldToObjectSpace( _pos_world_ );

    float3 _normal_object_  = ( _pos_object_ - make_float3( q ) ) / q.w;
    float3 _normal_world_   = normalize( optixTransformNormalFromObjectToWorldSpace( _normal_object_ ) );

    float3 P = _pos_world_;
    float3 N = _normal_world_;

    prd->geometryNormal = N;

    attrs.pos = P;
    attrs.nrm = N;
    attrs.uv = sphereUV(_normal_object_);

    attrs.clr = {};
    attrs.tang = {};
    attrs.instPos = {}; //rt_data->instPos[inst_idx2];
    attrs.instNrm = {}; //rt_data->instNrm[inst_idx2];
    attrs.instUv = {}; //rt_data->instUv[inst_idx2];
    attrs.instClr = {}; //rt_data->instClr[inst_idx2];
    attrs.instTang = {}; //rt_data->instTang[inst_idx2];

#else

    int inst_idx2 = optixGetInstanceIndex();
    int inst_idx = rt_data->meshIdxs[inst_idx2];
    int vert_idx_offset = (inst_idx * TRI_PER_MESH + prim_idx)*3;

    float m16[16];
    m16[12]=0; m16[13]=0; m16[14]=0; m16[15]=1;
    optixGetObjectToWorldTransformMatrix(m16);
    mat4& meshMat = *reinterpret_cast<mat4*>(&m16);

    float3 _vertices_[3];
    optixGetTriangleVertexData( gas,
                                prim_idx,
                                sbtGASIndex,
                                0,
                                _vertices_);
    
    float3 av0 = _vertices_[0]; //make_float3(rt_data->vertices[vert_idx_offset + 0]);
    float3 av1 = _vertices_[1]; //make_float3(rt_data->vertices[vert_idx_offset + 1]);
    float3 av2 = _vertices_[2]; //make_float3(rt_data->vertices[vert_idx_offset + 2]);
    vec4 bv0 = vec4(av0.x, av0.y, av0.z, 1);
    vec4 bv1 = vec4(av1.x, av1.y, av1.z, 1);
    vec4 bv2 = vec4(av2.x, av2.y, av2.z, 1);
    bv0 = meshMat * bv0;
    bv1 = meshMat * bv1;
    bv2 = meshMat * bv2;
    float3 v0 = make_float3(bv0.x, bv0.y, bv0.z);
    float3 v1 = make_float3(bv1.x, bv1.y, bv1.z);
    float3 v2 = make_float3(bv2.x, bv2.y, bv2.z);

    float3 N_0 = normalize( cross( normalize(v1-v0), normalize(v2-v1) ) ); // this value has precision issue for big float 
    
    if (isBadVector(N_0)) 
    {  
        N_0 = DisneyBSDF::SampleScatterDirection(prd->seed);
        N_0 = faceforward( N_0, -ray_dir, N_0 );
    }
    
    prd->geometryNormal = N_0;

    float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    unsigned short isLight = rt_data->lightMark[inst_idx * TRI_PER_MESH + prim_idx];
    //float w = rt_data->vertices[ vert_idx_offset+0 ].w;

    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();
    
//    float3 n0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
//
//    float3 n1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
//
//    float3 n2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));

    float3 uv0 = make_float3(rt_data->uv[ vert_idx_offset+0 ] );
    float3 uv1 = make_float3(rt_data->uv[ vert_idx_offset+1 ] );
    float3 uv2 = make_float3(rt_data->uv[ vert_idx_offset+2 ] );
    float3 clr0 = make_float3(rt_data->clr[ vert_idx_offset+0 ] );
    float3 clr1 = make_float3(rt_data->clr[ vert_idx_offset+1 ] );
    float3 clr2 = make_float3(rt_data->clr[ vert_idx_offset+2 ] );
    float3 atan0 = make_float3(rt_data->tan[ vert_idx_offset+0 ] );
    float3 atan1 = make_float3(rt_data->tan[ vert_idx_offset+1 ] );
    float3 atan2 = make_float3(rt_data->tan[ vert_idx_offset+2 ] );
    mat3 meshMat3x3(meshMat);
    vec3 btan0(atan0);
    vec3 btan1(atan1);
    vec3 btan2(atan2);
    btan0 = meshMat3x3 * btan0;
    btan1 = meshMat3x3 * btan1;
    btan2 = meshMat3x3 * btan2;
    float3 tan0 = make_float3(btan0.x, btan0.y, btan0.z);
    float3 tan1 = make_float3(btan1.x, btan1.y, btan1.z);
    float3 tan2 = make_float3(btan2.x, btan2.y, btan2.z);
    
    //N_0 = normalize(interp(barys, n0, n1, n2));
    float3 N = N_0;//faceforward( N_0, -ray_dir, N_0 );
    P = interp(barys, v0, v1, v2); // this value has precision issue for big float 
    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = N;
    attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = normalize(interp(barys, tan0, tan1, tan2));
    attrs.instPos = rt_data->instPos[inst_idx2];
    attrs.instNrm = rt_data->instNrm[inst_idx2];
    attrs.instUv = rt_data->instUv[inst_idx2];
    attrs.instClr = rt_data->instClr[inst_idx2];
    attrs.instTang = rt_data->instTang[inst_idx2];

#endif

    MatOutput mats = evalMaterial(zenotex, rt_data->uniforms, attrs);

#if _SPHERE_

    if(mats.doubleSide>0.5f||mats.thin>0.5f){
        N = faceforward( N, -ray_dir, N );
        prd->geometryNormal = N;
    }

#else

    float3 an0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
    vec3 bn0(an0);
    bn0 = meshMat3x3 * bn0;
    float3 n0 = make_float3(bn0.x, bn0.y, bn0.z);
    n0 = dot(n0, N_0)>(1-mats.smoothness)?n0:N_0;

    float3 an1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
    vec3 bn1(an1);
    bn1 = meshMat3x3 * bn1;
    float3 n1 = make_float3(bn1.x, bn1.y, bn1.z);
    n1 = dot(n1, N_0)>(1-mats.smoothness)?n1:N_0;

    float3 an2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));
    vec3 bn2(an2);
    bn2 = meshMat3x3 * bn2;
    float3 n2 = make_float3(bn2.x, bn2.y, bn2.z);
    n2 = dot(n2, N_0)>(1-mats.smoothness)?n2:N_0;
    N_0 = normalize(interp(barys, n0, n1, n2));
    N = N_0;

    if(mats.doubleSide>0.5f||mats.thin>0.5f){
        N = faceforward( N_0, -ray_dir, N_0 );
        prd->geometryNormal = faceforward( prd->geometryNormal, -ray_dir, prd->geometryNormal );
    }
#endif

    attrs.nrm = N;
    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01f,0.99f);
    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
    }

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

    /* MODME */
    auto basecolor = mats.basecolor;
    auto metallic = mats.metallic;
    auto roughness = mats.roughness;
    if(prd->diffDepth>=1)
        roughness = clamp(roughness, 0.2,0.99);
    if(prd->diffDepth>=2)
        roughness = clamp(roughness, 0.3,0.99);
    if(prd->diffDepth>=3)
        roughness = clamp(roughness, 0.5,0.99);

    auto subsurface = mats.subsurface;
    auto specular = mats.specular;
    auto specularTint = mats.specularTint;
    auto anisotropic = mats.anisotropic;
    auto anisoRotation = mats.anisoRotation;
    auto sheen = mats.sheen;
    auto sheenTint = mats.sheenTint;
    auto clearcoat = mats.clearcoat;
    auto clearcoatGloss = mats.clearcoatGloss;
    auto ccRough = mats.clearcoatRoughness;
    auto ccIor = mats.clearcoatIOR;
    auto opacity = mats.opacity;
    auto flatness = mats.flatness;
    auto specTrans = mats.specTrans;
    auto scatterDistance = mats.scatterDistance;
    auto ior = mats.ior;
    auto thin = mats.thin;

    auto sssColor = mats.sssColor;
    auto sssParam = mats.sssParam;

    auto scatterStep = mats.scatterStep;
    //discard fully opacity pixels
    //opacity = clamp(opacity, 0.0f, 0.99f);
    prd->opacity = opacity;
    if(prd->isSS == true) {
        basecolor = vec3(1.0f);
        roughness = 1.0;
        anisotropic = 0;
        sheen = 0;
        clearcoat = 0;
        specTrans = 0;
        ior = 1;
    }
    if(prd->isSS == true && subsurface>0 && dot(-normalize(ray_dir), N)>0)
    {
       prd->attenuation2 = make_float3(0,0,0);
       prd->attenuation = make_float3(0,0,0);
       prd->radiance = make_float3(0,0,0);
       prd->done = true;
       return;
    }
    if(prd->isSS == true  && subsurface==0 )
    {
        prd->passed = true;
        prd->samplePdf = 1.0f;
        prd->radiance = make_float3(0.0f, 0.0f, 0.0f);
        prd->opacity = 0;
        prd->readMat(prd->sigma_t, prd->ss_alpha);
        auto trans = DisneyBSDF::Transmission2(prd->sigma_s(), prd->sigma_t, prd->channelPDF, optixGetRayTmax(), true);
        prd->attenuation2 *= trans;
        prd->attenuation *= trans;
        //prd->origin = P + 1e-5 * ray_dir; 
        if(prd->maxDistance>optixGetRayTmax())
            prd->maxDistance-=optixGetRayTmax();
        prd->offsetUpdateRay(P, ray_dir); 
        return;
    }

    prd->attenuation2 = prd->attenuation;
    prd->countEmitted = false;
    if(isLight==1)
    {
        prd->countEmitted = true;
        //hit light, emit
//        float dist = length(P - optixGetWorldRayOrigin()) + 1e-5;
//        float3 lv1 = v1-v0;
//        float3 lv2 = v2-v0;
//        float A = 0.5 * length(cross(lv1, lv2));
//        float3 lnrm = normalize(cross(normalize(lv1), normalize(lv2)));
//        float3 L     = normalize(P - optixGetWorldRayOrigin());
//        float  LnDl  = clamp(-dot( lnrm, L ), 0.0f, 1.0f);
//        float weight = LnDl * A / (M_PIf * dist);
//        prd->radiance = attrs.clr * weight;
        prd->offsetUpdateRay(P, ray_dir); 
        return;
    }
    prd->prob2 = prd->prob;
    prd->passed = false;
    if(opacity>0.99f)
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
        //prd->samplePdf = 0.0f;
        prd->radiance = make_float3(0.0f);
        //prd->origin = P + 1e-5 * ray_dir; 
        prd->offsetUpdateRay(P, ray_dir);
        return;
    }
    if(opacity<=0.99f)
    {
      //we have some simple transparent thing
      //roll a dice to see if just pass
      if(rnd(prd->seed)<opacity)
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
        //prd->samplePdf = 0.0f;
        //you shall pass!
        prd->radiance = make_float3(0.0f);

        prd->origin = P;
        prd->direction = ray_dir;
        prd->offsetUpdateRay(P, ray_dir);

        prd->prob *= 1;
        prd->countEmitted = false;
        return;
      }
    }
    if(prd->depth==0&&flatness>0.5)
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
                basecolor,
                sssParam,
                sssColor,
                metallic,
                subsurface,
                specular,
                roughness,
                specularTint,
                anisotropic,
                anisoRotation,
                sheen,
                sheenTint,
                clearcoat,
                clearcoatGloss,
                ccRough,
                ccIor,
                flatness,
                specTrans,
                scatterDistance,
                ior,
                T,
                B,
                N,
                prd->geometryNormal,
                -normalize(ray_dir),
                thin>0.5f,
                prd->next_ray_is_going_inside,
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
        prd->next_ray_is_going_inside = dot(vec3(prd->geometryNormal),vec3(wi))<=0;
    }

    if(thin>0.5f || mats.doubleSide>0.5f)
    {
        if (prd->curMatIdx > 0) {
            vec3 sigma_t, ss_alpha;
            prd->readMat(sigma_t, ss_alpha);
            if (ss_alpha.x<0.0f) { // is inside Glass
                prd->attenuation *= DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                prd->attenuation2 *= DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
            } else {
                prd->attenuation *= DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
                prd->attenuation2 *= DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
            }
        }else {
            prd->attenuation *= 1;
        }
        prd->next_ray_is_going_inside = false;
    }else{
    
        //if(flag == DisneyBSDF::transmissionEvent || flag == DisneyBSDF::diracEvent) {
        if(istransmission || flag == DisneyBSDF::diracEvent) {
            if(prd->next_ray_is_going_inside){
                if(thin < 0.5f && mats.doubleSide < 0.5f ) 
                {
                    outToIn = true;
                    inToOut = false;

                    prd->medium = DisneyBSDF::PhaseFunctions::isotropic;

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
                    prd->channelPDF = vec3(1.0f/3.0f);
                    if (isTrans) {
                        vec3 channelPDF = vec3(1.0f/3.0f);
                        prd->maxDistance = scatterStep>0.5f? DisneyBSDF::SampleDistance2(prd->seed, prd->sigma_t, prd->sigma_t, channelPDF) : 1e16f;
                        prd->pushMat(extinction);
                    } else {

                        vec3 channelPDF = vec3(1.0f/3.0f);
                        prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * prd->ss_alpha, prd->sigma_t, channelPDF);
                        //here is the place caused inf ray:fixed
                        auto min_sg = fmax(fmin(fmin(prd->sigma_t.x, prd->sigma_t.y), prd->sigma_t.z), 1e-8f);
                        //what should be the right value???
                        //prd->maxDistance = max(prd->maxDistance, 10/min_sg);
                        //printf("maxdist:%f\n",prd->maxDistance);
                        prd->channelPDF = channelPDF;
                        // already calculated in BxDF

                        // if (idx.x == w/2 && idx.y == h/2) {
                        //     printf("into sss, sigma_t, alpha: %f, %f, %f\n", prd->sigma_t.x, prd->sigma_t.y, prd->sigma_t.z,prd->ss_alpha.x, prd->ss_alpha.y, prd->ss_alpha.z);
                        // }
                        
                        prd->pushMat(prd->sigma_t, prd->ss_alpha);
                    }

                    prd->scatterDistance = scatterDistance;
                    prd->scatterStep = scatterStep;
                }
                
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

                prd->attenuation2 *= trans;
                prd->attenuation *= trans;

                prd->popMat(sigma_t, ss_alpha);

                prd->medium = (prd->curMatIdx==0)? DisneyBSDF::PhaseFunctions::vacuum : DisneyBSDF::PhaseFunctions::isotropic;

                if(ss_alpha.x >= 0.0f) //next ray in 3s object
                {
                    prd->isSS = true;
                    prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * ss_alpha, sigma_t, prd->channelPDF);
                }
                else
                {
                    prd->isSS = false;
                    prd->maxDistance = 1e16;
                }

                // if (prd->medium != DisneyBSDF::PhaseFunctions::vacuum) {

                //     prd->bad = true;
                    
                //     printf("%f %f %f %f %f %f %f %f \n matIdx = %d isotropic = %d \n", prd->sigma_t_queue[0].x, prd->sigma_t_queue[1].x, prd->sigma_t_queue[2].x, prd->sigma_t_queue[3].x, prd->sigma_t_queue[4].x, prd->sigma_t_queue[5].x, prd->sigma_t_queue[6].x, prd->sigma_t_queue[7].x,
                //         prd->curMatIdx, prd->medium);
                //     printf("matIdx = %d isotropic = %d \n\n", prd->curMatIdx, prd->medium);
                // }
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
                    else if (prd->ss_alpha.x<0.0f) { // Glass
                        trans = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                        vec3 channelPDF = vec3(1.0f/3.0f);
                        prd->maxDistance = scatterStep>0.5f? DisneyBSDF::SampleDistance2(prd->seed, sigma_t, sigma_t, channelPDF) : 1e16f;
                    } else { // SSS
                        trans = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
                        prd->channelPDF = vec3(1.0f/3.0f);
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
    prd->medium = prd->next_ray_is_going_inside?DisneyBSDF::PhaseFunctions::isotropic : prd->curMatIdx==0?DisneyBSDF::PhaseFunctions::vacuum : DisneyBSDF::PhaseFunctions::isotropic;
 
    if(thin>0.5f){
        vec3 H = normalize(vec3(normalize(wi)) + vec3(-normalize(ray_dir)));
        attrs.N = N;
        attrs.T = cross(B,N);
        attrs.L = vec3(normalize(wi));
        attrs.V = vec3(-normalize(ray_dir));
        attrs.H = normalize(H);
        attrs.reflectance = reflectance;
        attrs.fresnel = DisneyBSDF::DisneyFresnel( basecolor, metallic, ior, specularTint, dot(attrs.H, attrs.V), dot(attrs.H, attrs.L), false);
        MatOutput mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
        reflectance = mat2.reflectance;
    }

    prd->countEmitted = false;
    prd->attenuation *= reflectance;
    prd->depth++;

    auto shadingP = rtgems::offset_ray(P,  prd->geometryNormal);
    prd->radiance = make_float3(0.0f,0.0f,0.0f);

    if(prd->depth>=3)
        roughness = clamp(roughness, 0.5f,0.99f);

    vec3 rd, rs, rt; // captured by lambda

    auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF, vec3 illum = vec3(1.0f)) -> float3 {

        const auto& L = _wi_; // pre-normalized
        const vec3& V = _wo_; // pre-normalized

        float3 lbrdf = DisneyBSDF::EvaluateDisney2(illum,
            basecolor, sssColor, metallic, subsurface, specular, max(prd->minSpecRough,roughness), specularTint, anisotropic, anisoRotation, sheen, sheenTint,
            clearcoat, clearcoatGloss, ccRough, ccIor, specTrans, scatterDistance, ior, flatness, L, V, T, B, N,prd->geometryNormal,
            thin > 0.5f, flag == DisneyBSDF::transmissionEvent ? inToOut : prd->next_ray_is_going_inside, thisPDF, rrPdf,
            dot(N, L), rd, rs, rt);

        MatOutput mat2;
        if(thin>0.5f){
            vec3 H = normalize(vec3(normalize(L)) + V);
            attrs.N = N;
            attrs.T = cross(B,N);
            attrs.L = vec3(normalize(L));
            attrs.V = V;
            attrs.H = normalize(H);
            attrs.reflectance = lbrdf;
            attrs.fresnel = DisneyBSDF::DisneyFresnel( basecolor, metallic, ior, specularTint, dot(attrs.H, attrs.V), dot(attrs.H, attrs.L), false);
            mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
        }

        return (thin>0.5f? float3(mat2.reflectance):lbrdf);
    };

    auto taskAux = [&](const vec3& weight) {
        prd->radiance_d = rd * weight;
        prd->radiance_s = rs * weight;
        prd->radiance_t = rt * weight;
    };

    RadiancePRD shadow_prd {};
    shadow_prd.seed = prd->seed;
    shadow_prd.shadowAttanuation = make_float3(1.0f, 1.0f, 1.0f);
    shadow_prd.nonThinTransHit = (thin == false && specTrans > 0) ? 1 : 0;

    prd->direction = normalize(wi);

    DirectLighting<true>(prd, shadow_prd, shadingP, ray_dir, evalBxDF, &taskAux);
    
    if(thin<0.5f && mats.doubleSide<0.5f){
        prd->origin = rtgems::offset_ray(P, (prd->next_ray_is_going_inside)? -prd->geometryNormal : prd->geometryNormal);
    }
    else {
        prd->origin = rtgems::offset_ray(P, ( dot(prd->direction, prd->geometryNormal) < 0 )? -prd->geometryNormal : prd->geometryNormal);
    }

    prd->radiance += float3(mats.emission);
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
