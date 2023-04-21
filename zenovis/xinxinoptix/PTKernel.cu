#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "DisneyBSDF.h"
#include "zxxglslvec.h"

#include <cuda_fp16.h>

extern "C" {
__constant__ Params params;

}
//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------
static __inline__ __device__
vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786f) - 0.000090537f;
    vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}
static __inline__ __device__
vec3 ACESFitted(vec3 color, float gamma)
{
//    const mat3x3 ACESInputMat = mat3x3
//        (
//            0.59719, 0.35458, 0.04823,
//            0.07600, 0.90834, 0.01566,
//            0.02840, 0.13383, 0.83777
//        );
//    mat3x3 ACESOutputMat = mat3x3
//    (
//        1.60475, -0.53108, -0.07367,
//        -0.10208,  1.10813, -0.00605,
//        -0.00327, -0.07276,  1.07602
//    );
    vec3 v1 = vec3(0.59719, 0.35458, 0.04823);
    vec3 v2 = vec3(0.07600, 0.90834, 0.01566);
    vec3 v3 = vec3(0.02840, 0.13383, 0.83777);
    color = vec3(dot(color, v1), dot(color, v2), dot(color, v3));
    // Apply RRT and ODT
    color = RRTAndODTFit(color);

    v1 = vec3(1.60475, -0.53108, -0.07367);
    v2 = vec3(-0.10208,  1.10813, -0.00605);
    v3 = vec3(-0.00327, -0.07276,  1.07602);
    color = vec3(dot(color, v1), dot(color, v2), dot(color, v3));

    // Clamp to [0, 1]
    color = clamp(color, 0.0, 1.0);

    color = pow(color, vec3(1. / gamma));

    return color;
}

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    //const float3 eye = params.eye;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;
    const CameraInfo cam = params.cam;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );
    float focalPlaneDistance = cam.focalPlaneDistance>0.01? cam.focalPlaneDistance : 0.01;
    float aperture = clamp(cam.aperture,0.0f,100.0f);
    aperture/=10;

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;

    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        float2 subpixel_jitter = {
            rnd(seed),
            rnd(seed)
        };

        float2 d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        //float3 ray_direction = normalize(cam.right * d.x + cam.up * d.y + cam.front);
        float2 r01 = {
            rnd(seed),
            rnd(seed)
        };
        
        float r0 = r01.x * 2.0f* M_PIf;
        float r1 = r01.y * aperture * aperture;
        r1 = sqrt(r1);

        // float3 ray_origin    = cam.eye + r1 * ( cosf(r0)* cam.right + sinf(r0)* cam.up);
        // float3 ray_direction = cam.eye + focalPlaneDistance *(cam.right * d.x + cam.up * d.y + cam.front) - ray_origin;
   
        float3 eye_shake     = r1 * ( cosf(r0)* cam.right + sinf(r0)* cam.up); // Camera local space

        float3 ray_origin    = cam.eye + eye_shake;
        float3 ray_direction = focalPlaneDistance *(cam.right * d.x + cam.up * d.y + cam.front) - eye_shake; // Camera local space
               ray_direction = normalize(ray_direction);

        RadiancePRD prd; 
        prd.emission     = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.attenuation2 = make_float3(1.f);
        prd.prob         = 1.0f;
        prd.prob2        = 1.0f;
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.opacity      = 0;
        prd.flags        = 0;
        prd.next_ray_is_going_inside    = false;
        prd.maxDistance  = 1e16f;
        prd.medium       = DisneyBSDF::PhaseFunctions::vacuum;

        prd.depth = 0;
        prd.diffDepth = 0;
        prd.isSS = false;
        prd.direction = ray_direction;
        prd.curMatIdx = 0;
        prd.test_distance = false;

        auto tmin = prd.trace_tmin;
        auto ray_mask = prd._mask_;

        // prd.channelPDF= vec3(1.0f/3.0f);
        // prd.ss_alpha = vec3(0.0f);
        // prd.sigma_t = vec3(0.0f);

        for(;;)
        {
            traceRadianceMasked(params.handle, ray_origin, ray_direction, tmin, prd.maxDistance, ray_mask, &prd);

            tmin = prd.trace_tmin;
            prd.trace_tmin = 0;

            ray_mask = prd._mask_; 
            prd._mask_ = EverythingMask;

//            vec3 radiance = vec3(prd.radiance);
//            vec3 oldradiance = radiance;
//            RadiancePRD shadow_prd;
//            shadow_prd.depth = prd.depth;
//            shadow_prd.shadowAttanuation = make_float3(1.0f, 1.0f, 1.0f);
//            shadow_prd.nonThinTransHit = prd.nonThinTransHit;
//            traceOcclusion(params.handle, prd.LP, prd.Ldir,
//                           1e-5f, // tmin
//                           1e16f, // tmax,
//                           &shadow_prd);
//            radiance = radiance * prd.Lweight * vec3(shadow_prd.shadowAttanuation);
//            radiance = radiance + vec3(prd.emission);

//            prd.radiance = float3(mix(oldradiance, radiance, prd.CH));

            // if (prd.bad) {
            //     result = make_float3(999, 0, 0);
            //     i = 1; break;
            // }

            if(prd.countEmitted==false || prd.depth>0) {
                result += prd.radiance * prd.attenuation2/(prd.prob2 + 1e-5);
                // fire without smoke requires this line to work.
            }

            prd.radiance = make_float3(0);
            prd.emission = make_float3(0);

            if (ray_mask != EverythingMask && ray_mask != NothingMask) {
                //ray_origin = prd.origin;
                //ray_direction = prd.direction;
                continue; // trace again with same parameters but different mask
            }

            if(prd.countEmitted==true && prd.depth>0){
                prd.done = true;
            }

            if( prd.done || params.simpleRender==true){
                break;
            }

            if(prd.depth>16){
                //float RRprob = clamp(length(prd.attenuation)/1.732f,0.01f,0.9f);
                float RRprob = clamp(length(prd.attenuation),0.1, 0.95);
                if(rnd(prd.seed) > RRprob || prd.depth>32){
                    prd.done=true;
                } else {
                    prd.attenuation = prd.attenuation / (RRprob + 1e-5);
                }
            }
            if(prd.countEmitted == true)
                prd.passed = true;

            ray_origin    = prd.origin;
            ray_direction = prd.direction;
            // if(prd.passed == false)
            //     ++depth;        
            //}else{
                //prd.passed = false;
            //}
        }
    }
    while( --i );

    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );
    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }

    /*if (launch_index.x == 0) {*/
        /*printf("%p\n", params.accum_buffer);*/
        /*printf("%p\n", params.frame_buffer);*/
    /*}*/
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    //vec3 aecs_fitted = ACESFitted(vec3(accum_color), 2.2);
    float3 out_color = accum_color;
    params.frame_buffer[ image_index ] = make_color ( out_color );
}

extern "C" __global__ void __miss__radiance()
{
    vec3 sunLightDir = vec3(
            params.sunLightDirX,
            params.sunLightDirY,
            params.sunLightDirZ
            );
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();
    prd->attenuation2 = prd->attenuation;
    prd->passed = false;
    prd->countEmitted = false;
    prd->CH = 0.0;
    if(prd->medium != DisneyBSDF::PhaseFunctions::isotropic){
        prd->radiance = envSky(
            normalize(prd->direction),
            sunLightDir,
            make_float3(0., 0., 1.),
            40, // be careful
            .45,
            15.,
            1.030725 * 0.3,
            params.elapsedTime
        );
        prd->done      = true;
        return;
    }

    vec3 transmittance = DisneyBSDF::Transmission2(prd->sigma_s(), prd->sigma_t, prd->channelPDF, optixGetRayTmax(), false);
    prd->attenuation *= transmittance;//DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->attenuation2 *= transmittance;//DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->origin += prd->direction * optixGetRayTmax();
    prd->direction = DisneyBSDF::SampleScatterDirection(prd->seed);

    vec3 channelPDF = vec3(1.0/3.0);
    prd->maxDistance = DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * prd->ss_alpha, prd->sigma_t, channelPDF);
    prd->channelPDF= channelPDF;
    prd->depth++;

    if(length(prd->attenuation)<1e-7f){
        prd->done = true;
    }
}

extern "C" __global__ void __miss__occlusion()
{
    setPayloadOcclusion( false );
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}