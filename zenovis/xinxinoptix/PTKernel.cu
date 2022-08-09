#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "DisneyBSDF.h"

extern "C" {
__constant__ Params params;
}
//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w   = params.width;
    const int    h   = params.height;
    //const float3 eye = params.eye;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;
    const CameraInfo cam = params.cam;

    unsigned int seed = tea<4>( idx.y*w + idx.x, subframe_index );

    float3 result = make_float3( 0.0f );
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        float2 subpixel_jitter = make_float2( rnd( seed ), rnd( seed ) );

        float2 d = 2.0f * make_float2(
                ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
                ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
                ) - 1.0f;
        float3 ray_direction = normalize(cam.right * d.x + cam.up * d.y + cam.front);
        float3 ray_origin    = cam.eye;

        RadiancePRD prd;
        prd.emitted      = make_float3(0.f);
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
        prd.is_inside    = false;
        prd.maxDistance  = 1e16f;
        prd.medium       = DisneyBSDF::PhaseFunctions::vacuum;
        int depth = 0;
        for( ;; )
        {
            traceRadiance(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    1e-5f,  // tmin       // TODO: smarter offset
                    prd.maxDistance,  // tmax
                    &prd );

            //result += prd.emitted;
            if(prd.countEmitted==false || depth>0)
                result += prd.radiance * prd.attenuation2/(prd.prob2+1e-5);
            if(prd.countEmitted==true && depth>0){
                prd.done = true;
            }
            if( prd.done ){ 
                break;
            }
            if(depth>5){
                float RRprob = clamp(length((prd.attenuation)/1.732f),0.01f,0.99f);
                    // float RRprob = prd.prob;
                if(rnd(prd.seed) < RRprob){
                    //prd.attenuation = make_float3(0.0f);
                    break;
                }
                prd.attenuation = prd.attenuation / RRprob;
            }


            if(prd.countEmitted == true)
                prd.passed = true;
            ray_origin    = prd.origin;
            ray_direction = prd.direction;
            if(prd.passed == false)
                ++depth;
        }
    }
    while( --i );

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index  = launch_index.y * params.width + launch_index.x;
    float3         accum_color  = result / static_cast<float>( params.samples_per_launch );

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
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    RadiancePRD* prd = getPRD();

    if(prd->medium != DisneyBSDF::isotropic){
        prd->radiance = make_float3( rt_data->bg_color );
        prd->done      = true;
    }
    prd->attenuation *= DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->origin += prd->direction * optixGetRayTmax();
    prd->direction = DisneyBSDF::SampleScatterDirection(prd->seed);
    float tmpPDF;
    prd->maxDistance = DisneyBSDF::SampleDistance(prd->seed,prd->extinction,tmpPDF);
    prd->scatterPDF= tmpPDF;

    if(length(prd->attenuation)<1e-5f){
        prd->done = true;
    }

}
