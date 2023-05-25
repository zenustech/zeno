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

extern "C" __global__ void __anyhit__shadow_cutout()
{
    RadiancePRD* prd = getPRD();
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();

    int inst_idx2 = optixGetInstanceIndex();
    int inst_idx = rt_data->meshIdxs[inst_idx2];
    int vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;

    float* meshMats = rt_data->meshMats;
    mat4 meshMat = mat4(
        meshMats[16 * inst_idx2 + 0], meshMats[16 * inst_idx2 + 1], meshMats[16 * inst_idx2 + 2], meshMats[16 * inst_idx2 + 3],
        meshMats[16 * inst_idx2 + 4], meshMats[16 * inst_idx2 + 5], meshMats[16 * inst_idx2 + 6], meshMats[16 * inst_idx2 + 7],
        meshMats[16 * inst_idx2 + 8], meshMats[16 * inst_idx2 + 9], meshMats[16 * inst_idx2 + 10], meshMats[16 * inst_idx2 + 11],
        meshMats[16 * inst_idx2 + 12], meshMats[16 * inst_idx2 + 13], meshMats[16 * inst_idx2 + 14], meshMats[16 * inst_idx2 + 15]);
    float3 av0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);
    float3 av1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
    float3 av2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
    vec4 bv0 = vec4(av0.x, av0.y, av0.z, 1);
    vec4 bv1 = vec4(av1.x, av1.y, av1.z, 1);
    vec4 bv2 = vec4(av2.x, av2.y, av2.z, 1);
    bv0 = meshMat * bv0;
    bv1 = meshMat * bv1;
    bv2 = meshMat * bv2;
    float3 v0 = make_float3(bv0.x, bv0.y, bv0.z);
    float3 v1 = make_float3(bv1.x, bv1.y, bv1.z);
    float3 v2 = make_float3(bv2.x, bv2.y, bv2.z);

    float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    
    auto zenotex = rt_data->textures;

    MatInput attrs;
    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();
    
    mat3 meshMat3x3(meshMat);
    float3 an0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
    vec3 bn0(an0);
    bn0 = meshMat3x3 * bn0;
    float3 n0 = make_float3(bn0.x, bn0.y, bn0.z);
    n0 = dot(n0, N_0)>0.8?n0:N_0;

    float3 an1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
    vec3 bn1(an1);
    bn1 = meshMat3x3 * bn1;
    float3 n1 = make_float3(bn1.x, bn1.y, bn1.z);
    n1 = dot(n1, N_0)>0.8?n1:N_0;

    float3 an2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));
    vec3 bn2(an2);
    bn2 = meshMat3x3 * bn2;
    float3 n2 = make_float3(bn2.x, bn2.y, bn2.z);
    n2 = dot(n2, N_0)>0.8?n2:N_0;
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
    MatOutput mats = evalMaterial(zenotex, rt_data->uniforms, attrs);

    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
    }
    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01,0.99);

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
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    if(params.simpleRender==true)
        opacity = 0;
    //opacity = clamp(opacity, 0.0f, 0.99f);
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99 || isLight == 1) // No need to calculate an expensive random number if the test is going to fail anyway.
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
            if(length(prd->shadowAttanuation) < 0.01){
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
int GetLightIndex(float p, ParallelogramLight* lightP, int n)
{
    int s = 0, e = n-1;
    while( s < e )
    {
        int j = (s+e)/2;
        float pc = lightP[j].cdf/lightP[n-1].cdf;
        if(pc<p)
        {
            s = j+1;
        }
        else
        {
            e = j;
        }
    }
    return e;
}
static __inline__ __device__
vec3 projectedBarycentricCoord(vec3 p, vec3 q, vec3 u, vec3 v)
{
    vec3 n = cross(u,v);
    float a = 1.0 / dot(n,n);
    vec3 w = p - q;
    vec3 o;
    o.z = dot(cross(u,w),n) * a;
    o.y = dot(cross(w,v),n) * a;
    o.x = 1.0 - o.y - o.z;
    return o;
}
vec3 ImportanceSampleEnv(float* env_cdf, int nx, int ny, float p, float &pdf)
{
    int start = 0; int end = nx*ny-1;
    while(start<end-1)
    {
        int mid = (start + end)/2;
        if(env_cdf[mid]<p)
        {
            start = mid;
        }
        else
        {
            end = mid;
        }
    }
    int i = start%nx;
    int j = start/nx;
    float theta = ((float)i + 0.5f)/(float) nx * 2.0f * 3.1415926f - 3.1415926f;
    float phi = ((float)j + 0.5f)/(float) ny * 3.1415926f;
    float twoPi2sinTheta = 2.0f * M_PIf * M_PIf * sin(phi);
    pdf =  twoPi2sinTheta / env_cdf[start + nx*ny];
    return normalize(vec3(cos(theta), sin(phi - 0.5 * 3.1415926f), sin(theta)));

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

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    int    prim_idx        = optixGetPrimitiveIndex();
    float3 ray_dir         = optixGetWorldRayDirection();

    int inst_idx2 = optixGetInstanceIndex();
    int inst_idx = rt_data->meshIdxs[inst_idx2];
    int vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;

    float* meshMats = rt_data->meshMats;
    mat4 meshMat = mat4(
        meshMats[16 * inst_idx2 + 0], meshMats[16 * inst_idx2 + 1], meshMats[16 * inst_idx2 + 2], meshMats[16 * inst_idx2 + 3],
        meshMats[16 * inst_idx2 + 4], meshMats[16 * inst_idx2 + 5], meshMats[16 * inst_idx2 + 6], meshMats[16 * inst_idx2 + 7],
        meshMats[16 * inst_idx2 + 8], meshMats[16 * inst_idx2 + 9], meshMats[16 * inst_idx2 + 10], meshMats[16 * inst_idx2 + 11],
        meshMats[16 * inst_idx2 + 12], meshMats[16 * inst_idx2 + 13], meshMats[16 * inst_idx2 + 14], meshMats[16 * inst_idx2 + 15]);
    float3 av0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);
    float3 av1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
    float3 av2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
    vec4 bv0 = vec4(av0.x, av0.y, av0.z, 1);
    vec4 bv1 = vec4(av1.x, av1.y, av1.z, 1);
    vec4 bv2 = vec4(av2.x, av2.y, av2.z, 1);
    bv0 = meshMat * bv0;
    bv1 = meshMat * bv1;
    bv2 = meshMat * bv2;
    float3 v0 = make_float3(bv0.x, bv0.y, bv0.z);
    float3 v1 = make_float3(bv1.x, bv1.y, bv1.z);
    float3 v2 = make_float3(bv2.x, bv2.y, bv2.z);

    float3 N_0  = normalize( cross( v1-v0, v2-v1 ) );
        prd->geometryNormal = N_0;

    float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    float w = rt_data->vertices[ vert_idx_offset+0 ].w;

    auto zenotex = rt_data->textures;

    MatInput attrs;
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
    P = interp(barys, v0, v1, v2);
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

    MatOutput mats = evalMaterial(zenotex, rt_data->uniforms, attrs);
    
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
    if(mats.doubleSide>0.5||mats.thin>0.5){
        N = faceforward( N_0, -ray_dir, N_0 );
        prd->geometryNormal = faceforward( prd->geometryNormal, -ray_dir, prd->geometryNormal );
    }
    attrs.nrm = N;
    //end of material computation
    //mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01,0.99);
    auto N2 = N;
    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
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

    if(prd->isSS == true  && subsurface==0 )
    {
        prd->passed = true;
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
    if(opacity>0.99)
    {
        prd->passed = true;
        prd->radiance = make_float3(0.0f);
        //prd->origin = P + 1e-5 * ray_dir; 
        prd->offsetUpdateRay(P, ray_dir);
        return;
    }

    
    float is_refl;
    float3 inDir = ray_dir;
    vec3 wi = vec3(0.0f);
    float pdf = 0.0f;
    float rPdf = 0.0f;
    float fPdf = 0.0f;
    float rrPdf = 0.0f;
    float ffPdf = 0.0f;
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

    while(DisneyBSDF::SampleDisney(
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
                isTrans
                )  == false)
        {
            isSS = false;
            isDiff = false;
            rPdf = 0.0f;
            fPdf = 0.0f;
            reflectance = vec3(0.0f);
            flag = DisneyBSDF::scatterEvent;
        }
    prd->isSS = isSS;
    pdf = fPdf;
    if(isDiff || prd->diffDepth>0){
        prd->diffDepth++;
    }
    
    if(opacity<=0.99)
    {
        //we have some simple transparent thing
        //roll a dice to see if just pass
        if(rnd(prd->seed)<opacity)
        {
            prd->passed = true;
            //you shall pass!
            prd->radiance = make_float3(0.0f);

            prd->origin = P;
            prd->direction = ray_dir;
            prd->offsetUpdateRay(P, ray_dir); 

            prd->prob *= 1;
            prd->countEmitted = false;
            prd->attenuation *= 1;
            return;
        }
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

    if(thin>0.5 || mats.doubleSide>0.5)
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
                if(thin < 0.5 && mats.doubleSide < 0.5 ) 
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
                        vec3 channelPDF = vec3(1.0/3.0);
                        prd->maxDistance = scatterStep>0.5? DisneyBSDF::SampleDistance2(prd->seed, prd->sigma_t, prd->sigma_t, channelPDF) : 1e16;
                        prd->pushMat(extinction);
                    } else {

                        vec3 channelPDF = vec3(1.0/3.0);
                        prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * prd->ss_alpha, prd->sigma_t, channelPDF);
                        //here is the place caused inf ray:fixed
                        auto min_sg = max(min(min(prd->sigma_t.x, prd->sigma_t.y), prd->sigma_t.z), 1e-8);
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
                        prd->maxDistance = 1e16;
                    }
                    else if (prd->ss_alpha.x<0.0f) { // Glass
                        trans = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
                        vec3 channelPDF = vec3(1.0/3.0);
                        prd->maxDistance = scatterStep>0.5? DisneyBSDF::SampleDistance2(prd->seed, sigma_t, sigma_t, channelPDF) : 1e16;
                    } else { // SSS
                        trans = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), true);
                        prd->channelPDF = vec3(1.0/3.0);
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
 
    if(thin>0.5){
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

    auto P_OLD = P;
    P = rtgems::offset_ray(P,  prd->geometryNormal);

    prd->radiance = make_float3(0.0f,0.0f,0.0f);
    float3 light_attenuation = make_float3(1.0f,1.0f,1.0f);
    float pl = rnd(prd->seed);
    int lidx = GetLightIndex(pl, params.lights, params.num_lights);
    float sum = 0.0f;
    for(int lidx=0;lidx<params.num_lights;lidx++)
    {
            ParallelogramLight light = params.lights[lidx];
            float3 light_pos = light.corner + light.v1 * 0.5 + light.v2 * 0.5;

            // Calculate properties of light sample (for area based pdf)
            float Ldist = length(light_pos - P);
            float3 L = normalize(light_pos - P);
            float nDl = 1.0f;//clamp(dot(N, L), 0.0f, 1.0f);
            float LnDl = clamp(-dot(light.normal, L), 0.000001f, 1.0f);
            float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            sum += length(light.emission)  * nDl * LnDl * A / (M_PIf * Ldist * Ldist );

    }
    if(prd->depth>=3)
        roughness = clamp(roughness, 0.5,0.99);

    RadiancePRD shadow_prd {};
    shadow_prd.seed = prd->seed;
    shadow_prd.shadowAttanuation = make_float3(1.0f, 1.0f, 1.0f);
    shadow_prd.nonThinTransHit = (thin == false && specTrans > 0) ? 1 : 0;

    if(rnd(prd->seed)<=0.5) {
        bool computed = false;
        float ppl = 0;
        for (int lidx = 0; lidx < params.num_lights && computed == false; lidx++) {
            ParallelogramLight light = params.lights[lidx];
            float2 z = {rnd(prd->seed), rnd(prd->seed)};
            const float z1 = z.x;
            const float z2 = z.y;
            float3 light_tpos = light.corner + light.v1 * 0.5 + light.v2 * 0.5;
            float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

            // Calculate properties of light sample (for area based pdf)
            float tLdist = length(light_tpos - P);
            float3 tL = normalize(light_tpos - P);
            float tnDl = 1.0f; //clamp(dot(N, tL), 0.0f, 1.0f);
            float tLnDl = clamp(-dot(light.normal, tL), 0.000001f, 1.0f);
            float tA = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            ppl += length(light.emission) * tnDl * tLnDl * tA / (M_PIf * tLdist * tLdist) / sum;
            if (ppl > pl) {
                float Ldist = length(light_pos - P) + 1e-6;
                float3 L = normalize(light_pos - P);
                float nDl = 1.0f; //clamp(dot(N, L), 0.0f, 1.0f);
                float LnDl = clamp(-dot(light.normal, L), 0.0f, 1.0f);
                float A = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
                float weight = 0.0f;
                if (nDl > 0.0f && LnDl > 0.0f) {

                    traceOcclusion(params.handle, P, L,
                                   1e-5f,         // tmin
                                   Ldist - 1e-5f, // tmax,
                                   &shadow_prd);

                    light_attenuation = shadow_prd.shadowAttanuation;
                    if (fmaxf(light_attenuation) > 0.0f) {

                        weight = sum * nDl / tnDl * LnDl / tLnDl * (tLdist * tLdist) / (Ldist  * Ldist) /
                                 (length(light.emission)+1e-6f) ;
                    }
                }
                prd->LP = P;
                prd->Ldir = L;
                prd->nonThinTransHit = (thin == false && specTrans > 0) ? 1 : 0;
                prd->Lweight = weight;

                float3 lbrdf = DisneyBSDF::EvaluateDisney(
                    basecolor, sssColor, metallic, subsurface, specular, roughness, specularTint, anisotropic, anisoRotation, sheen, sheenTint,
                    clearcoat, clearcoatGloss, ccRough, ccIor, specTrans, scatterDistance, ior, flatness, L, -normalize(inDir), T, B, N,
                    thin > 0.5f, flag == DisneyBSDF::transmissionEvent ? inToOut : prd->next_ray_is_going_inside, ffPdf, rrPdf,
                    dot(N, L));
                MatOutput mat2;
                if(thin>0.5){
                    vec3 H = normalize(vec3(normalize(L)) + vec3(-normalize(inDir)));
                    attrs.N = N;
                    attrs.T = cross(B,N);
                    attrs.L = vec3(normalize(L));
                    attrs.V = vec3(-normalize(inDir));
                    attrs.H = normalize(H);
                    attrs.reflectance = lbrdf;
                    attrs.fresnel = DisneyBSDF::DisneyFresnel( basecolor, metallic, ior, specularTint, dot(attrs.H, attrs.V), dot(attrs.H, attrs.L), false);
                    mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
                }
                prd->radiance = light_attenuation * weight * 2.0 * light.emission * (thin>0.5? float3(mat2.reflectance):lbrdf);
                computed = true;
            }
        }
    } else {
    for(int samples=0;samples<20;samples++) {
        float3 lbrdf{};
        bool inside = false;
        float p = rnd(prd->seed);
        //vec3 sunLightDir = vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);
        float envpdf = 0;
        vec3 sunLightDir = ImportanceSampleEnv(params.skycdf, params.skynx, params.skyny, p, envpdf);
        auto sun_dir = BRDFBasics::halfPlaneSample(prd->seed, sunLightDir,
                                                   params.sunSoftness * 0); //perturb the sun to have some softness
        sun_dir = normalize(sunLightDir);
        float3 illum = float3(envSky(sun_dir, sun_dir, make_float3(0., 0., 1.),
                                     40, // be careful
                                     .45, 15., 1.030725 * 0.3, params.elapsedTime));
        prd->LP = P;
        prd->Ldir = sun_dir;
        prd->nonThinTransHit = (thin == false && specTrans > 0) ? 1 : 0;
        prd->Lweight = 1.0;

        traceOcclusion(params.handle, P, sun_dir,
                       1e-5f, // tmin
                       1e16f, // tmax,
                       &shadow_prd);
        lbrdf = DisneyBSDF::EvaluateDisney(
            basecolor, sssColor, metallic, subsurface, specular, roughness, specularTint, anisotropic,
            anisoRotation, sheen, sheenTint, clearcoat, clearcoatGloss, ccRough, ccIor, specTrans, scatterDistance,
            ior, flatness, sun_dir, -normalize(inDir), T, B, N, thin > 0.5f,
            flag == DisneyBSDF::transmissionEvent ? inToOut : prd->next_ray_is_going_inside, ffPdf, rrPdf,
            dot(N, float3(sun_dir)));
        light_attenuation = shadow_prd.shadowAttanuation;
        //if (fmaxf(light_attenuation) > 0.0f) {
        //            auto sky = float3(envSky(sun_dir, sunLightDir, make_float3(0., 0., 1.),
        //                                          10, // be careful
        //                                          .45, 15., 1.030725 * 0.3, params.elapsedTime));
        MatOutput mat2;
        if (thin > 0.5) {
            vec3 H = normalize(vec3(normalize(sun_dir)) + vec3(-normalize(inDir)));
            attrs.N = N;
            attrs.T = cross(B, N);
            attrs.L = vec3(normalize(sun_dir));
            attrs.V = vec3(-normalize(inDir));
            attrs.H = normalize(H);
            attrs.reflectance = lbrdf;
            attrs.fresnel = DisneyBSDF::DisneyFresnel(basecolor, metallic, ior, specularTint, dot(attrs.H, attrs.V),
                                                      dot(attrs.H, attrs.L), false);
            mat2 = evalReflectance(zenotex, rt_data->uniforms, attrs);
        }
        prd->radiance += 1.0f/20.0f  *
            light_attenuation * illum * envpdf * 2.0 * (thin > 0.5 ? float3(mat2.reflectance) : lbrdf);
    }
    }

    P = P_OLD;
    prd->direction = normalize(wi);
    if(thin<0.5 && mats.doubleSide<0.5){
        prd->origin = rtgems::offset_ray(P, (prd->next_ray_is_going_inside)? -prd->geometryNormal : prd->geometryNormal);
    }
    else {
        prd->origin = rtgems::offset_ray(P, ( dot(prd->direction, prd->geometryNormal) <0 )? -prd->geometryNormal : prd->geometryNormal);
    }

    

    prd->radiance +=  float3(mats.emission);
    prd->CH = 1.0;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
