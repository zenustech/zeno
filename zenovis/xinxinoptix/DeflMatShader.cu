#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "DisneyBRDF.h"
#include "IOMat.h"



/*
extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
extern "C" __global__ void __anyhit__shadow_cutout()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const vec3 ray_dir         = (optixGetWorldRayDirection());
    const int    vert_idx_offset = prim_idx*3;

    const vec3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const vec3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const vec3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const vec3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const vec3 N    = faceforward( N_0, -ray_dir, N_0 );
    const vec3 P    = vec3(optixGetWorldRayOrigin()) + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();
    float opacity = 0.0;//sin(P.y)>0?1.0:0.0;
    prd->opacity = opacity;
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99 ) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
    }
    else
    {
        prd->flags |= 1;
        optixTerminateRay();
    }
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const vec3 ray_dir         = (optixGetWorldRayDirection());
    const int    vert_idx_offset = prim_idx*3;

    const vec3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const vec3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const vec3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const vec3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const vec3 N    = faceforward( N_0, -ray_dir, N_0 );
    const vec3 P    = vec3(optixGetWorldRayOrigin()) + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    if( prd->countEmitted )
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = vec3( 0.0f );

    
    float opacity = 0.0;//sin(P.y)>0?1.0:0.0;
    prd->opacity = opacity;
    if(opacity>0.99)
    {
        prd->radiance += vec3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }

    unsigned int seed = prd->seed;

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        vec3 w_in;
        cosine_sample_hemisphere( z1, z2, w_in );
        Onb onb( N );
        onb.inverse_transform( w_in );
        prd->direction = w_in;
        prd->origin    = P;

        prd->attenuation *= rt_data->diffuse_color;
        prd->countEmitted = false;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.light;
    const vec3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const vec3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( vec3(light.normal), L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        prd->flags = 0;
        traceOcclusion(
            params.handle,
            P,
            L,
            1e-5f,         // tmin
            Ldist - 1e-5f  // tmax
            );
        unsigned int occluded = prd->flags;
        if( !occluded )
        {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd->radiance += light.emission * weight;
}
*/


static __inline__ __device__ MatOutput evalMaterial(MatInput const &attrs) {
    /* MODMA */
    auto attr_pos = attrs.pos;
    auto attr_clr = attrs.clr;
    auto attr_uv = attrs.uv;
    auto attr_nrm = attrs.nrm;
    auto attr_tang = attrs.tang;
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
    return mats;
}


extern "C" __global__ void __anyhit__shadow_cutout()
{
    RadiancePRD* prd = getPRD();
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    const int    prim_idx        = optixGetPrimitiveIndex();
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    inst_idx        = 0;//optixGetInstanceIndex();
    const int    vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;
    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );

    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    
    MatInput attrs;
    /* MODMA */
    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = vec3(0,0,1);
    attrs.uv = vec3(0,0,0);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = vec3(1,1,1);
    attrs.tang = vec3(0,0,0);
    MatOutput mats = evalMaterial(attrs);
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
    if(isLight==1)
        mats.opacity = 1;
    // Stochastic alpha test to get an alpha blend effect.
    if (opacity >0.99) // No need to calculate an expensive random number if the test is going to fail anyway.
    {
        optixIgnoreIntersection();
    }
    else
    {
        //roll a dice
        float p = rnd(prd->seed);
        if(p<opacity)
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
    const int    inst_idx        = 0;//optixGetInstanceIndex();
    const int    vert_idx_offset = (inst_idx * 1024 + prim_idx)*3;
    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );
    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    unsigned short isLight = rt_data->lightMark[inst_idx * 1024 + prim_idx];
    float w = rt_data->vertices[ vert_idx_offset+0 ].w;
    MatInput attrs;
    /* MODMA */
    attrs.pos = vec3(P.x, P.y, P.z);
    attrs.nrm = vec3(0,0,1);
    attrs.uv = vec3(0,0,0);//todo later
    //attrs.clr = rt_data->face_attrib_clr[vert_idx_offset];
    attrs.clr = vec3(1,1,1);
    MatOutput mats = evalMaterial(attrs);
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
        prd->radiance = make_float3(1.0f,1.0f,1.0f) * w * weight;
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }
    if(opacity>0.99)
    {
        prd->radiance = make_float3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }
    
    //{
    unsigned int seed = prd->seed;
    float is_refl;
    float3 inDir = ray_dir;
    float3 wi = DisneyBRDF::sample_f(
                                seed,
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

    if(opacity<0.99)
    {
        //we have some simple transparent thing
        //roll a dice to see if just pass
        float ratioTrans = 0.5 * opacity;
        float p = rnd(seed);
        pdf = (1-ratioTrans) * pdf + ratioTrans * 1;
        if(p<ratioTrans)
        {
            //you shall pass!
            prd->radiance = make_float3(0.0f);
            prd->origin = P;
            prd->direction = ray_dir;
            prd->prob *= pdf;
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

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.lights[0];
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = clamp(dot( N, L ),0.0f,1.0f);
    const float  LnDl  = clamp(-dot( light.normal, L ),0.0f,1.0f);

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        prd->flags = 0;
        traceOcclusion(
            params.handle,
            P,
            L,
            1e-5f,         // tmin
            Ldist - 1e-5f  // tmax
            );
        unsigned int occluded = prd->flags;
        if( !occluded )
        {
            float wpdf = DisneyBRDF::pdf(basecolor,
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
                                L,
                                -normalize(inDir)
                                );
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }
    float3 lbrdf = DisneyBRDF::eval(basecolor,
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
                                L,
                                -normalize(inDir)
                                );
    prd->radiance = light.emission * weight * lbrdf ;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
