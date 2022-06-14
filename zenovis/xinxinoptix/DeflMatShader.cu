#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "DisneyBRDF.h"
#include "IOMat.h"

//COMMON_CODE


static __inline__ __device__ MatOutput evalMaterial(
cudaTextureObject_t zenotex0 , 
cudaTextureObject_t zenotex1 , 
cudaTextureObject_t zenotex2 , 
cudaTextureObject_t zenotex3 , 
cudaTextureObject_t zenotex4 , 
cudaTextureObject_t zenotex5 , 
cudaTextureObject_t zenotex6 , 
cudaTextureObject_t zenotex7 , 
cudaTextureObject_t zenotex8 , 
cudaTextureObject_t zenotex9 , 
cudaTextureObject_t zenotex10, 
cudaTextureObject_t zenotex11, 
cudaTextureObject_t zenotex12, 
cudaTextureObject_t zenotex13, 
cudaTextureObject_t zenotex14, 
cudaTextureObject_t zenotex15, 
cudaTextureObject_t zenotex16, 
cudaTextureObject_t zenotex17, 
cudaTextureObject_t zenotex18, 
cudaTextureObject_t zenotex19, 
cudaTextureObject_t zenotex20, 
cudaTextureObject_t zenotex21, 
cudaTextureObject_t zenotex22, 
cudaTextureObject_t zenotex23, 
cudaTextureObject_t zenotex24, 
cudaTextureObject_t zenotex25, 
cudaTextureObject_t zenotex26, 
cudaTextureObject_t zenotex27, 
cudaTextureObject_t zenotex28, 
cudaTextureObject_t zenotex29, 
cudaTextureObject_t zenotex30, 
cudaTextureObject_t zenotex31, 
MatInput const &attrs) {
    /* MODMA */
    auto att_pos = attrs.pos;
    auto att_clr = attrs.clr;
    auto att_uv = attrs.uv;
    auto att_nrm = attrs.nrm;
    auto att_tang = attrs.tang;
    /** generated code here beg **/
    //GENERATED_BEGIN_MARK
    /* MODME */
    vec3 mat_basecolor = vec3(1.0, 0.0, 1.0);
    float mat_metallic = 0.0;
    float mat_roughness = 0.5;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearcoat = 0.0;
    float mat_clearcoatGloss = 0.0;
    float mat_opacity = 0.0;
    vec3  mat_normal = vec3(0.0f, 0.0f, 1.0f);
    vec3 mat_emission = vec3(0.0f, 0.0f,0.0f);
    //GENERATED_END_MARK
    /** generated code here end **/
    MatOutput mats;
    /* MODME */
    mats.basecolor = mat_basecolor;
    mats.metallic = mat_metallic;
    mats.roughness = mat_roughness;
    mats.subsurface = mat_subsurface;
    mats.specular = mat_specular;
    mats.specularTint = mat_specularTint;
    mats.anisotropic = mat_anisotropic;
    mats.sheen = mat_sheen;
    mats.sheenTint = mat_sheenTint;
    mats.clearcoat = mat_clearcoat;
    mats.clearcoatGloss = mat_clearcoatGloss;
    mats.opacity = mat_opacity;
    mats.nrm = mat_normal;
    mats.emission = mat_emission;
    return mats;
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
    const int    inst_idx        = optixGetInstanceIndex();
    const int    vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;
    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );

    float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    cudaTextureObject_t zenotex0  = rt_data->textures[0 ];
    cudaTextureObject_t zenotex1  = rt_data->textures[1 ];
    cudaTextureObject_t zenotex2  = rt_data->textures[2 ];
    cudaTextureObject_t zenotex3  = rt_data->textures[3 ];
    cudaTextureObject_t zenotex4  = rt_data->textures[4 ];
    cudaTextureObject_t zenotex5  = rt_data->textures[5 ];
    cudaTextureObject_t zenotex6  = rt_data->textures[6 ];
    cudaTextureObject_t zenotex7  = rt_data->textures[7 ];
    cudaTextureObject_t zenotex8  = rt_data->textures[8 ];
    cudaTextureObject_t zenotex9  = rt_data->textures[9 ];
    cudaTextureObject_t zenotex10 = rt_data->textures[10];
    cudaTextureObject_t zenotex11 = rt_data->textures[11];
    cudaTextureObject_t zenotex12 = rt_data->textures[12];
    cudaTextureObject_t zenotex13 = rt_data->textures[13];
    cudaTextureObject_t zenotex14 = rt_data->textures[14];
    cudaTextureObject_t zenotex15 = rt_data->textures[15];
    cudaTextureObject_t zenotex16 = rt_data->textures[16];
    cudaTextureObject_t zenotex17 = rt_data->textures[17];
    cudaTextureObject_t zenotex18 = rt_data->textures[18];
    cudaTextureObject_t zenotex19 = rt_data->textures[19];
    cudaTextureObject_t zenotex20 = rt_data->textures[20];
    cudaTextureObject_t zenotex21 = rt_data->textures[21];
    cudaTextureObject_t zenotex22 = rt_data->textures[22];
    cudaTextureObject_t zenotex23 = rt_data->textures[23];
    cudaTextureObject_t zenotex24 = rt_data->textures[24];
    cudaTextureObject_t zenotex25 = rt_data->textures[25];
    cudaTextureObject_t zenotex26 = rt_data->textures[26];
    cudaTextureObject_t zenotex27 = rt_data->textures[27];
    cudaTextureObject_t zenotex28 = rt_data->textures[28];
    cudaTextureObject_t zenotex29 = rt_data->textures[29];
    cudaTextureObject_t zenotex30 = rt_data->textures[30];
    cudaTextureObject_t zenotex31 = rt_data->textures[31];
    MatInput attrs;
    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();
    
    float3 n0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
    n0 = dot(n0, N_0)>0.8?n0:N_0;

    float3 n1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
    n1 = dot(n1, N_0)>0.8?n1:N_0;

    float3 n2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));
    n2 = dot(n2, N_0)>0.8?n2:N_0;
    float3 uv0 = make_float3(rt_data->uv[ vert_idx_offset+0 ] );
    float3 uv1 = make_float3(rt_data->uv[ vert_idx_offset+1 ] );
    float3 uv2 = make_float3(rt_data->uv[ vert_idx_offset+2 ] );
    float3 clr0 = make_float3(rt_data->clr[ vert_idx_offset+0 ] );
    float3 clr1 = make_float3(rt_data->clr[ vert_idx_offset+1 ] );
    float3 clr2 = make_float3(rt_data->clr[ vert_idx_offset+2 ] );
    float3 tan0 = make_float3(rt_data->tan[ vert_idx_offset+0 ] );
    float3 tan1 = make_float3(rt_data->tan[ vert_idx_offset+1 ] );
    float3 tan2 = make_float3(rt_data->tan[ vert_idx_offset+2 ] );
    
    N_0 = normalize(interp(barys, n0, n1, n2));
    float3 N    = faceforward( N_0, -ray_dir, N_0 );

    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = N;
    attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = interp(barys, tan0, tan1, tan2);
    MatOutput mats = evalMaterial(
                                zenotex0 , 
                                zenotex1 , 
                                zenotex2 , 
                                zenotex3 , 
                                zenotex4 , 
                                zenotex5 , 
                                zenotex6 , 
                                zenotex7 , 
                                zenotex8 , 
                                zenotex9 , 
                                zenotex10, 
                                zenotex11, 
                                zenotex12, 
                                zenotex13, 
                                zenotex14, 
                                zenotex15, 
                                zenotex16, 
                                zenotex17, 
                                zenotex18, 
                                zenotex19, 
                                zenotex20, 
                                zenotex21, 
                                zenotex22, 
                                zenotex23, 
                                zenotex24, 
                                zenotex25, 
                                zenotex26, 
                                zenotex27, 
                                zenotex28, 
                                zenotex29, 
                                zenotex30, 
                                zenotex31,attrs);
    if(length(attrs.tang)>0)
    {
        vec3 b = cross(attrs.tang, attrs.nrm);
        attrs.tang = cross(attrs.nrm, b);
        N = mats.nrm.x * attrs.tang + mats.nrm.y * b + mats.nrm.z * attrs.nrm;
    }
    //end of material computation
    mats.metallic = clamp(mats.metallic,0.01, 0.99);
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
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    if(isLight==1)
        opacity = 1;
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
    }
    else
    {

        //roll a dice
        float p = rnd(prd->seed);
        if (p < opacity)
            optixIgnoreIntersection();
        prd->flags |= 1;
        optixTerminateRay();
    }
}


extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    inst_idx        = optixGetInstanceIndex();
    const int    vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;
    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    cudaTextureObject_t zenotex0  = rt_data->textures[0 ];
    cudaTextureObject_t zenotex1  = rt_data->textures[1 ];
    cudaTextureObject_t zenotex2  = rt_data->textures[2 ];
    cudaTextureObject_t zenotex3  = rt_data->textures[3 ];
    cudaTextureObject_t zenotex4  = rt_data->textures[4 ];
    cudaTextureObject_t zenotex5  = rt_data->textures[5 ];
    cudaTextureObject_t zenotex6  = rt_data->textures[6 ];
    cudaTextureObject_t zenotex7  = rt_data->textures[7 ];
    cudaTextureObject_t zenotex8  = rt_data->textures[8 ];
    cudaTextureObject_t zenotex9  = rt_data->textures[9 ];
    cudaTextureObject_t zenotex10 = rt_data->textures[10];
    cudaTextureObject_t zenotex11 = rt_data->textures[11];
    cudaTextureObject_t zenotex12 = rt_data->textures[12];
    cudaTextureObject_t zenotex13 = rt_data->textures[13];
    cudaTextureObject_t zenotex14 = rt_data->textures[14];
    cudaTextureObject_t zenotex15 = rt_data->textures[15];
    cudaTextureObject_t zenotex16 = rt_data->textures[16];
    cudaTextureObject_t zenotex17 = rt_data->textures[17];
    cudaTextureObject_t zenotex18 = rt_data->textures[18];
    cudaTextureObject_t zenotex19 = rt_data->textures[19];
    cudaTextureObject_t zenotex20 = rt_data->textures[20];
    cudaTextureObject_t zenotex21 = rt_data->textures[21];
    cudaTextureObject_t zenotex22 = rt_data->textures[22];
    cudaTextureObject_t zenotex23 = rt_data->textures[23];
    cudaTextureObject_t zenotex24 = rt_data->textures[24];
    cudaTextureObject_t zenotex25 = rt_data->textures[25];
    cudaTextureObject_t zenotex26 = rt_data->textures[26];
    cudaTextureObject_t zenotex27 = rt_data->textures[27];
    cudaTextureObject_t zenotex28 = rt_data->textures[28];
    cudaTextureObject_t zenotex29 = rt_data->textures[29];
    cudaTextureObject_t zenotex30 = rt_data->textures[30];
    cudaTextureObject_t zenotex31 = rt_data->textures[31];
    MatInput attrs;
    /* MODMA */
    float2       barys    = optixGetTriangleBarycentrics();
    
    float3 n0 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+0 ] ));
    n0 = dot(n0, N_0)>0.8?n0:N_0;

    float3 n1 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+1 ] ));
    n1 = dot(n1, N_0)>0.8?n1:N_0;

    float3 n2 = normalize(make_float3(rt_data->nrm[ vert_idx_offset+2 ] ));
    n2 = dot(n2, N_0)>0.8?n2:N_0;

    float3 uv0 = make_float3(rt_data->uv[ vert_idx_offset+0 ] );
    float3 uv1 = make_float3(rt_data->uv[ vert_idx_offset+1 ] );
    float3 uv2 = make_float3(rt_data->uv[ vert_idx_offset+2 ] );
    float3 clr0 = make_float3(rt_data->clr[ vert_idx_offset+0 ] );
    float3 clr1 = make_float3(rt_data->clr[ vert_idx_offset+1 ] );
    float3 clr2 = make_float3(rt_data->clr[ vert_idx_offset+2 ] );
    float3 tan0 = make_float3(rt_data->tan[ vert_idx_offset+0 ] );
    float3 tan1 = make_float3(rt_data->tan[ vert_idx_offset+1 ] );
    float3 tan2 = make_float3(rt_data->tan[ vert_idx_offset+2 ] );
    
    N_0 = normalize(interp(barys, n0, n1, n2));
    float3 N    = faceforward( N_0, -ray_dir, N_0 );

    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = N;
    attrs.uv = interp(barys, uv0, uv1, uv2);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = interp(barys, clr0, clr1, clr2);
    attrs.tang = normalize(interp(barys, tan0, tan1, tan2));
    MatOutput mats = evalMaterial(
                                zenotex0 , 
                                zenotex1 , 
                                zenotex2 , 
                                zenotex3 , 
                                zenotex4 , 
                                zenotex5 , 
                                zenotex6 , 
                                zenotex7 , 
                                zenotex8 , 
                                zenotex9 , 
                                zenotex10, 
                                zenotex11, 
                                zenotex12, 
                                zenotex13, 
                                zenotex14, 
                                zenotex15, 
                                zenotex16, 
                                zenotex17, 
                                zenotex18, 
                                zenotex19, 
                                zenotex20, 
                                zenotex21, 
                                zenotex22, 
                                zenotex23, 
                                zenotex24, 
                                zenotex25, 
                                zenotex26, 
                                zenotex27, 
                                zenotex28, 
                                zenotex29, 
                                zenotex30, 
                                zenotex31,attrs);
    //end of material computation
    mats.metallic = clamp(mats.metallic,0.01, 0.99);
    mats.roughness = clamp(mats.roughness, 0.01,0.99);
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
    auto subsurface = mats.subsurface;
    auto specular = mats.specular;
    auto specularTint = mats.specularTint;
    auto anisotropic = mats.anisotropic;
    auto sheen = mats.sheen;
    auto sheenTint = mats.sheenTint;
    auto clearcoat = mats.clearcoat;
    auto clearcoatGloss = mats.clearcoatGloss;
    auto opacity = mats.opacity;

    //discard fully opacity pixels
    prd->opacity = opacity;
    prd->prob2 = prd->prob;
    prd->attenuation2 = prd->attenuation;
    prd->countEmitted = false;
    if(isLight==1)
    {
        prd->countEmitted = true;
        //hit light, emit
        float dist = length(P - optixGetWorldRayOrigin());
        float3 lv1 = v1-v0, lv2 = v2-v0;
        float3 lnrm = normalize(cross(normalize(lv1), normalize(lv2)));
        float3 L     = normalize(P - optixGetWorldRayOrigin());
        float  LnDl  = clamp(-dot( lnrm, L ),0.0f,1.0f);
        float A = length(cross(lv1, lv2))/2;
        float weight = LnDl * A / (M_PIf*dist * dist);
        prd->radiance = attrs.clr * weight;
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }
    prd->passed = false;
    if(opacity>0.99)
    {
        prd->passed = true;
        prd->radiance = make_float3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }

    
    
    //{
    
    float is_refl;
    float3 inDir = ray_dir;
    float3 wi = DisneyBRDF::sample_f(
                                prd->seed,
                                basecolor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearcoat,
                                clearcoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                -normalize(ray_dir),
                                is_refl);

    float pdf = DisneyBRDF::pdf(basecolor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearcoat,
                                clearcoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    float3 f = DisneyBRDF::eval(basecolor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearcoat,
                                clearcoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
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
            prd->prob *= 1;
            prd->countEmitted = false;
            prd->attenuation *= 1;
            return;

        }

    }
    
    prd->prob *= pdf/clamp(dot(wi, N),0.0f,1.0f);
    prd->origin = P;
    prd->direction = wi;
    prd->countEmitted = false;
    prd->attenuation *= f;
    //}

    // {
    //     const float z1 = rnd(seed);
    //     const float z2 = rnd(seed);

    //     float3 w_in;
    //     cosine_sample_hemisphere( z1, z2, w_in );
    //     Onb onb( N );
    //     onb.inverse_transform( w_in );
    //     prd->direction = w_in;
    //     prd->origin    = P;

    //     prd->attenuation *= rt_data->diffuse_color;
    //     prd->countEmitted = false;
    // }

    prd->radiance = make_float3(0,0,0);
    for(int lidx=0;lidx<params.num_lights;lidx++) {
        ParallelogramLight light = params.lights[lidx];
        float z1 = rnd(prd->seed);
        float z2 = rnd(prd->seed);
        const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // Calculate properties of light sample (for area based pdf)
        const float Ldist = length(light_pos - P);
        const float3 L = normalize(light_pos - P);
        const float nDl = clamp(dot(N, L), 0.0f, 1.0f);
        const float LnDl = clamp(-dot(light.normal, L), 0.0f, 1.0f);

        float weight = 0.0f;
        if (nDl > 0.0f && LnDl > 0.0f) {
            prd->flags = 0;
            traceOcclusion(params.handle, P, L,
                           1e-5f,        // tmin
                           Ldist - 1e-5f // tmax
            );
            unsigned int occluded = prd->flags;
            if (!occluded) {
                float wpdf = DisneyBRDF::pdf(basecolor, metallic, subsurface, specular, roughness, specularTint,
                                             anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss, N,
                                             make_float3(0, 0, 0), make_float3(0, 0, 0), L, -normalize(inDir));
                const float A = length(cross(light.v1, light.v2));
                weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
            }
        }
        float3 lbrdf = DisneyBRDF::eval(basecolor, metallic, subsurface, specular, roughness, specularTint, anisotropic,
                                        sheen, sheenTint, clearcoat, clearcoatGloss, N, make_float3(0, 0, 0),
                                        make_float3(0, 0, 0), L, -normalize(inDir));
        prd->radiance += light.emission * weight * lbrdf + float3(mats.emission);
    }
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
