#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "MaterialStuff.h"



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


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}
extern "C" __global__ void __anyhit__shadow_cutout()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx        = optixGetPrimitiveIndex();
    const vec3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const vec3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const vec3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const vec3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const vec3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const vec3 N    = faceforward( N_0, -ray_dir, N_0 );
    const vec3 P    = vec3(optixGetWorldRayOrigin()) + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    vec3  mat_baseColor = vec3(1.0,0.766,0.336);
    float mat_metallic = 1;
    float mat_roughness = 0.1;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearCoat = 0.0;
    float mat_clearCoatGloss = 0.0;
    float mat_opacity = 0.0;
    vec3 attr_pos = vec3(P.x, P.y, P.z);
    vec3 attr_norm = vec3(0,0,1);
    vec3 attr_uv = vec3(0,0,0);//todo later
    vec3 attr_clr = vec3(rt_data->diffuse_color.x, rt_data->diffuse_color.y, rt_data->diffuse_color.z);
    vec3 attr_tang = vec3(0,0,0);
///////here injecting of material code in GLSL style///////////////////////////////


    float pnoise = perlin(1, 3, attr_pos*0.02);
    pnoise = clamp(pnoise, 0.0f, 1.0f);

    float pnoise2 = perlin(1, 4, attr_pos*0.02);
    mat_metallic = pnoise;

    mat_roughness = pnoise2;
    mat_roughness = clamp(mat_roughness, 0.01f,0.99f)*0.5f;

    float pnoise3 = perlin(10.0, 5, attr_pos*0.005);
    mat_opacity = clamp(pnoise3, 0.0f,1.0f);

////////////end of GLSL material code injection///////////////////////////////////////////////
    vec3 baseColor = mat_baseColor;
    float metallic = mat_metallic;;
    float roughness = mat_roughness;
    float subsurface = mat_subsurface;
    float specular = mat_specular;
    float specularTint = mat_specularTint;
    float anisotropic = mat_anisotropic;
    float sheen = mat_sheen;
    float sheenTint = mat_sheenTint;
    float clearCoat = mat_clearCoat;
    float clearCoatGloss = mat_clearCoatGloss;
    float opacity = mat_opacity;
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
    const float3 ray_dir         = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx*3;

    const float3 v0   = make_float3( rt_data->vertices[ vert_idx_offset+0 ] );
    const float3 v1   = make_float3( rt_data->vertices[ vert_idx_offset+1 ] );
    const float3 v2   = make_float3( rt_data->vertices[ vert_idx_offset+2 ] );
    const float3 N_0  = normalize( cross( v1-v0, v2-v0 ) );

    const float3 N    = faceforward( N_0, -ray_dir, N_0 );
    const float3 P    = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;

    RadiancePRD* prd = getPRD();

    if( prd->countEmitted )
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = make_float3( 0.0f );


    vec3  mat_baseColor = vec3(1.0,0.766,0.336);
    float mat_metallic = 1;
    float mat_roughness = 0.1;
    float mat_subsurface = 0.0;
    float mat_specular = 0;
    float mat_specularTint = 0.0;
    float mat_anisotropic = 0.0;
    float mat_sheen = 0.0;
    float mat_sheenTint = 0.0;
    float mat_clearCoat = 0.0;
    float mat_clearCoatGloss = 0.0;
    float mat_opacity = 0.0;
    vec3 attr_pos = vec3(P.x, P.y, P.z);
    vec3 attr_norm = vec3(0,0,1);
    vec3 attr_uv = vec3(0,0,0);//todo later
    vec3 attr_clr = vec3(rt_data->diffuse_color.x, rt_data->diffuse_color.y, rt_data->diffuse_color.z);
    vec3 attr_tang = vec3(0,0,0);
///////here injecting of material code in GLSL style///////////////////////////////


    float pnoise = perlin(1, 3, attr_pos*0.02);
    pnoise = clamp(pnoise, 0.0f, 1.0f);

    float pnoise2 = perlin(1, 4, attr_pos*0.02);
    mat_metallic = pnoise;

    mat_roughness = pnoise2;
    mat_roughness = clamp(mat_roughness, 0.01f,0.99f)*0.5f;

    float pnoise3 = perlin(10.0, 5, attr_pos*0.005);
    mat_opacity = clamp(pnoise3, 0.0f,1.0f);

////////////end of GLSL code injection///////////////////////////////////////////////
    vec3 baseColor = mat_baseColor;
    float metallic = mat_metallic;;
    float roughness = mat_roughness;
    float subsurface = mat_subsurface;
    float specular = mat_specular;
    float specularTint = mat_specularTint;
    float anisotropic = mat_anisotropic;
    float sheen = mat_sheen;
    float sheenTint = mat_sheenTint;
    float clearCoat = mat_clearCoat;
    float clearCoatGloss = mat_clearCoatGloss;
    float opacity = mat_opacity;
    //todo normal mapping TBN*N;



    //end of material computation
    metallic = clamp(metallic,0.01, 0.99);
    roughness = clamp(roughness, 0.01,0.99);
    //discard fully opacity pixels
    prd->opacity = opacity;
    if(opacity>0.99)
    {
        prd->radiance += make_float3(0.0f);
        prd->origin = P;
        prd->direction = ray_dir;
        return;
    }

    //{
    unsigned int seed = prd->seed;
    float is_refl;
    float3 wi = DisneyBRDF::sample_f(
                                seed,
                                baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                -normalize(ray_dir),
                                is_refl);

    float pdf = DisneyBRDF::pdf(baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    float3 f = DisneyBRDF::eval(baseColor,
                                metallic,
                                subsurface,
                                specular,
                                roughness,
                                specularTint,
                                anisotropic,
                                sheen,
                                sheenTint,
                                clearCoat,
                                clearCoatGloss,
                                N,
                                make_float3(0,0,0),
                                make_float3(0,0,0),
                                wi,
                                -normalize(ray_dir)
                                );
    prd->prob2 = prd->prob;
    prd->prob *= pdf;
    prd->origin = P;
    prd->direction = wi;
    prd->countEmitted = false;
    if(is_refl)
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
    else
        prd->attenuation *= f * clamp(dot(wi, N),0.0f,1.0f);
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

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P );
    const float3 L     = normalize(light_pos - P );
    const float  nDl   = dot( N, L );
    const float  LnDl  = -dot( light.normal, L );

    float weight = 0.0f;
    if( nDl > 0.0f && LnDl > 0.0f )
    {
        prd->flags = 0;
        traceOcclusion(
            params.handle,
            P,
            L,
            0.01f,         // tmin
            Ldist - 0.01f  // tmax
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
