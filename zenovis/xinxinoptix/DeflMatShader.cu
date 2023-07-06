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

#define _SPHERE_ 0
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
    size_t inst_idx2 = optixGetInstanceIndex();
    size_t inst_idx = rt_data->meshIdxs[inst_idx2];
    size_t vert_idx_offset = (inst_idx * TRI_PER_MESH + prim_idx)*3;

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
    float a = 1.0f / dot(n,n);
    vec3 w = p - q;
    vec3 o;
    o.z = dot(cross(u,w),n) * a;
    o.y = dot(cross(w,v),n) * a;
    o.x = 1.0 - o.y - o.z;
    return o;
}

static __inline__ __device__
vec3 ImportanceSampleEnv(float* env_cdf, int* env_start, int nx, int ny, float p, float &pdf)
{
    if(nx*ny == 0)
    {
        pdf = 1.0f;
        return vec3(0);
    }
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
    start = env_start[start];
    int i = start%nx;
    int j = start/nx;
    float theta = ((float)i + 0.5f)/(float) nx * 2.0f * 3.1415926f - 3.1415926f;
    float phi = ((float)j + 0.5f)/(float) ny * 3.1415926f;
    float twoPi2sinTheta = 2.0f * M_PIf * M_PIf * sin(phi);
    pdf = env_cdf[start + nx*ny] / twoPi2sinTheta;
    vec3 dir = normalize(vec3(cos(theta), sin(phi - 0.5f * 3.1415926f), sin(theta)));
    dir = dir.rotY(to_radians(-params.sky_rot))
            .rotZ(to_radians(-params.sky_rot_z))
            .rotX(to_radians(-params.sky_rot_x))
            .rotY(to_radians(-params.sky_rot_y));
    return dir;

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

    size_t inst_idx2 = optixGetInstanceIndex();
    size_t inst_idx = rt_data->meshIdxs[inst_idx2];
    size_t vert_idx_offset = (inst_idx * TRI_PER_MESH + prim_idx)*3;

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
        roughness = clamp(roughness, 0.5f,0.99f);

    RadiancePRD shadow_prd {};
    shadow_prd.seed = prd->seed;
    shadow_prd.shadowAttanuation = make_float3(1.0f, 1.0f, 1.0f);
    shadow_prd.nonThinTransHit = (thin == false && specTrans > 0) ? 1 : 0;

    if(rnd(prd->seed)<=0.5f) {
        bool computed = false;
        float ppl = 0;
        for (int lidx = 0; lidx < params.num_lights && computed == false; lidx++) {
            ParallelogramLight light = params.lights[lidx];
            float2 z = {rnd(prd->seed), rnd(prd->seed)};
            const float z1 = z.x;
            const float z2 = z.y;
            float3 light_tpos = light.corner + light.v1 * 0.5f + light.v2 * 0.5f;
            float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

            // Calculate properties of light sample (for area based pdf)
            float tLdist = length(light_tpos - P);
            float3 tL = normalize(light_tpos - P);
            float tnDl = 1.0f; //clamp(dot(N, tL), 0.0f, 1.0f);
            float tLnDl = clamp(-dot(light.normal, tL), 0.000001f, 1.0f);
            float tA = length(cross(params.lights[lidx].v1, params.lights[lidx].v2));
            ppl += length(light.emission) * tnDl * tLnDl * tA / (M_PIf * tLdist * tLdist) / sum;
            if (ppl > pl) {
                float Ldist = length(light_pos - P) + 1e-6f;
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
                vec3 rd, rs, rt;
                float3 lbrdf = DisneyBSDF::EvaluateDisney2(vec3(1.0f),
                    basecolor, sssColor, metallic, subsurface, specular, max(prd->minSpecRough,roughness), specularTint, anisotropic, anisoRotation, sheen, sheenTint,
                    clearcoat, clearcoatGloss, ccRough, ccIor, specTrans, scatterDistance, ior, flatness, L, -normalize(inDir), T, B, N,prd->geometryNormal,
                    thin > 0.5f, flag == DisneyBSDF::transmissionEvent ? inToOut : prd->next_ray_is_going_inside, ffPdf, rrPdf,
                    dot(N, L), rd, rs, rt);
                MatOutput mat2;
                if(thin>0.5f){
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
                prd->radiance = light_attenuation * weight * 2.0f * light.emission * (thin>0.5f? float3(mat2.reflectance):lbrdf);
                computed = true;
            }
        }
    } else {
        float env_weight_sum = 1e-8f;
        int NSamples = prd->depth<=2?1:1;//16 / pow(4.0f, (float)prd->depth-1);
    for(int samples=0;samples<NSamples;samples++) {
        float3 lbrdf{};
        bool inside = false;
        float p = rnd(prd->seed);
        //vec3 sunLightDir = vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);
        int hasenv = params.skynx * params.skyny;
        hasenv = params.usingHdrSky? hasenv : 0;
        float envpdf = 1;
        float3 illum = make_float3(0,0,0);
        vec3 sunLightDir = hasenv? ImportanceSampleEnv(params.skycdf, params.sky_start,
                                                        params.skynx, params.skyny, p, envpdf)
                                  : vec3(params.sunLightDirX, params.sunLightDirY, params.sunLightDirZ);
        auto sun_dir = BRDFBasics::halfPlaneSample(prd->seed, sunLightDir,
                                                   params.sunSoftness * 0.0f); //perturb the sun to have some softness
        sun_dir = hasenv ? normalize(sunLightDir):normalize(sun_dir);
        float tmpPdf;
        illum = float3(envSky(sun_dir, sunLightDir, make_float3(0., 0., 1.),
                                     40, // be careful
                                     .45, 15., 1.030725f * 0.3f, params.elapsedTime, tmpPdf));

        auto LP = P;
        auto Ldir = sun_dir;

        rtgems::offset_ray(LP, sun_dir);
        traceOcclusion(params.handle, LP, sun_dir,
                       1e-5f, // tmin
                       1e16f, // tmax,
                       &shadow_prd);
        vec3 rd, rs, rt;
        lbrdf = DisneyBSDF::EvaluateDisney2(vec3(illum),
            basecolor, sssColor, metallic, subsurface, specular, roughness, specularTint, anisotropic,
            anisoRotation, sheen, sheenTint, clearcoat, clearcoatGloss, ccRough, ccIor, specTrans, scatterDistance,
            ior, flatness, sun_dir, -normalize(inDir), T, B, N, prd->geometryNormal,thin > 0.5f,
            flag == DisneyBSDF::transmissionEvent ? inToOut : prd->next_ray_is_going_inside, ffPdf, rrPdf,
            dot(N, float3(sun_dir)), rd, rs, rt);
        light_attenuation = shadow_prd.shadowAttanuation;
        //if (fmaxf(light_attenuation) > 0.0f) {
        //            auto sky = float3(envSky(sun_dir, sunLightDir, make_float3(0., 0., 1.),
        //                                          10, // be careful
        //                                          .45, 15., 1.030725 * 0.3, params.elapsedTime));
        MatOutput mat2;
        if (thin > 0.5f) {
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
        float misWeight = BRDFBasics::PowerHeuristic(envpdf, ffPdf);
        misWeight = misWeight>0.0f?misWeight:1.0f;
        misWeight = ffPdf>1e-5f?misWeight:0.0f;
        misWeight = envpdf>1e-5?misWeight:0.0f;
        prd->radiance += misWeight * 1.0f / (float)NSamples *
            light_attenuation  / envpdf * 2.0f * (thin > 0.5f ? float3(mat2.reflectance) : lbrdf);
        prd->radiance_d = rd * vec3(misWeight * 1.0f / (float)NSamples *
                          light_attenuation  / envpdf * 2.0f);
        prd->radiance_s = rs * vec3(misWeight * 1.0f / (float)NSamples *
                          light_attenuation  / envpdf * 2.0f);
        prd->radiance_t = rt * vec3(misWeight * 1.0f / (float)NSamples *
                          light_attenuation  / envpdf * 2.0f);
    }
        //prd->radiance = float3(clamp(vec3(prd->radiance), vec3(0.0f), vec3(100.0f)));
    }

    P = P_OLD;
    prd->direction = normalize(wi);
    if(thin<0.5f && mats.doubleSide<0.5f){
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
