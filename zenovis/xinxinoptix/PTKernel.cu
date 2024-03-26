#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>

#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "DisneyBSDF.h"
#include "zxxglslvec.h"
#include "proceduralSky.h"

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
vec3 ACESFilm(vec3 x)
{
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0), vec3(1));
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
    color = clamp(color, 0.0f, 1.0f);

    //color = pow(color, vec3(1.0f / gamma));

    return color;
}
static __inline__ __device__
vec3 HdrToLDR(vec3 in)
{
  vec3 mapped = in;//vec3(1.0f) - exp(-in * 1.0/32.0);
  //mapped = pow(mapped, 1.0f/2.2f);
  return mapped;
}
static __inline__ __device__
vec3 PhysicalCamera(vec3 in,
                   float aperture = 2,
                   float shutterSpeed = 1.0/25,
                   float iso = 150,
                   float middleGrey = 0.18f,
                   bool enableExposure = true,
                   bool enableACES = true)
{
  vec3 mapped;
  float exposure = middleGrey / ( (1000.0f / 65.0f) * aperture * aperture / (iso * shutterSpeed) );
  mapped = in * exposure;
  return  enableExposure? (enableACES? ACESFilm(mapped):mapped ) : (enableACES? ACESFilm(in) : in);
}
extern "C" __global__ void __raygen__rg()
{

    const int    w   = params.windowSpace.x;
    const int    h   = params.windowSpace.y;
    //const float3 eye = params.eye;
    const uint3  idxx = optixGetLaunchIndex();
    uint3 idx;
    idx.x = idxx.x + params.tile_i * params.tile_w;
    idx.y = idxx.y + params.tile_j * params.tile_h;
    if(idx.x>w || idx.y>h)
        return;

    const unsigned int image_index  = idx.y * w + idx.x;
    const int    subframe_index = params.subframe_index;
    const CameraInfo cam = params.cam;

    int seedy = idx.y/4, seedx = idx.x/8;
    int sid = (idx.y%4) * 8 + idx.x%8;
    unsigned int seed = tea<4>( idx.y * w + idx.x, subframe_index);
    unsigned int eventseed = tea<4>( idx.y * w + idx.x, subframe_index + 1);
    seed += params.outside_random_number;
    eventseed += params.outside_random_number;
    float focalPlaneDistance = cam.focal_distance>0.01f? cam.focal_distance: 0.01f;
    float aperture = clamp(cam.aperture,0.0f,100.0f);
    float physical_aperture = 0.0f;
    if(aperture < 0.05f || aperture > 24.0f){
        physical_aperture = 0.0f;
    }else{
        physical_aperture = cam.focal_length / aperture;
    }
    

    float3 result = make_float3( 0.0f );
    float3 result_d = make_float3( 0.0f );
    float3 result_s = make_float3( 0.0f );
    float3 result_t = make_float3( 0.0f );
    float3 result_b = make_float3( 0.0f );
    float3 aov[4];
    int i = params.samples_per_launch;

    float3 tmp_albedo{};
    float3 tmp_normal{};
    unsigned int sobolseed = subframe_index;
    float3 mask_value = make_float3( 0.0f );

    do{
        // The center of each pixel is at fraction (0.5,0.5)
        float2 subpixel_jitter = sobolRnd(sobolseed);

        float2 d = 2.0f * make_float2(
            ( static_cast<float>( idx.x + params.windowCrop_min.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
            ( static_cast<float>( idx.y + params.windowCrop_min.y ) + subpixel_jitter.y ) / static_cast<float>( h )
            ) - 1.0f;

        float2 r01 = sobolRnd(eventseed);

        float r0 = r01.x * 2.0f * M_PIf;
        float r1 = sqrtf(r01.y) * physical_aperture;

        float sin_yaw = sinf(cam.yaw);
        float cos_yaw = cosf(cam.yaw);
        float sin_pitch = sinf(cam.pitch);
        float cos_pitch = cosf(cam.pitch);

        mat3 tile_transform = mat3(
            cos_yaw, -sin_yaw * cos_pitch,  sin_pitch*sin_yaw,
            sin_yaw, cos_yaw * cos_pitch,   - cos_yaw * sin_pitch,
            0.0f,    sin_pitch,             cos_pitch
        );

        mat3 camera_transform = mat3(
            cam.right.x, cam.up.x, cam.front.x,
            cam.right.y, cam.up.y, cam.front.y,
            cam.right.z, cam.up.z, cam.front.z
        );

        // Under camer local space, cam.eye as origin, cam.right as X axis, cam.up as Y axis, cam.front as Z axis.
        float3 eye_shake     = r1 * (cosf(r0) * make_float3(1.0f,0.0f,0.0f) + sinf(r0) * make_float3(0.0f,1.0f,0.0f)); // r1 * ( cos(r0) , sin(r0) , 0 );
        float3 focal_plane_center = make_float3(cam.vertical_shift*cam.height, cam.horizontal_shift*cam.width, cam.focal_length);
        float3 old_direction =   focal_plane_center + make_float3(cam.width * 0.5f * d.x, cam.height * 0.5f * d.y, 0.0f);
        float3 tile_normal =  make_float3(sin_pitch*sin_yaw, - cos_yaw * sin_pitch, cos_pitch);

        float D = - dot(tile_normal , focal_plane_center);//surcface equaltion is Ax+By+Cz+D = 0 
        
        /*to sphere coordinate
        x = r * sin(theta) * cos(phi) = r * C1;
        y = r * sin(theta) * sin(phi) = r * C2;
        z = r * cos(phi) = r* C3;
        */
        float old_r = length(old_direction);
        float3 C_vector = old_direction/old_r;
        float new_r = -D / dot(tile_normal,C_vector); 
        /*
        Ax+By+Cz+D = A*C1*r + B*C2*r + C*C3*r + D = ((A,B,C) dot (C1,C2,C3)) * r + D =0
        old_direction/old_r = (C1,C2,C3)
        */
        float3 terminal_point = new_r * C_vector;
        terminal_point = terminal_point * (cam.focal_distance/cam.focal_length);//focal_length control

        //transform to world space
        terminal_point = camera_transform * terminal_point;
        eye_shake = camera_transform * eye_shake;

        float3 ray_origin    = eye_shake;
        float3 ray_direction = terminal_point - eye_shake; 
        ray_direction = normalize(ray_direction);

        RadiancePRD prd;
        prd.pixel_area   = cam.height/(float)(h)/(cam.focal_length);
        prd.adepth       = 0;
        prd.emission     = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.attenuation2 = make_float3(1.f);
        prd.prob         = 1.0f;
        prd.prob2        = 1.0f;
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.eventseed    = eventseed;
        prd.flags        = 0;
        prd.maxDistance  = 1e16f;
        prd.medium       = DisneyBSDF::PhaseFunctions::vacuum;

        prd.origin = ray_origin;
        prd.direction = ray_direction;
        prd.samplePdf = 1.0f;
        prd.mask_value = make_float3( 0.0f );

        prd.depth = 0;
        prd.diffDepth = 0;
        prd.isSS = false;
        prd.curMatIdx = 0;
        prd.test_distance = false;
        prd.ss_alpha_queue[0] = vec3(-1.0f);
        prd.minSpecRough = 0.01;
        prd.samplePdf = 1.0f;
        prd.hit_type = 0;
        prd.max_depth = 6;
        auto _tmin_ = prd._tmin_;
        auto _mask_ = prd._mask_;
        
        //if constexpr(params.denoise) 
        if (params.denoise) 
        {
            prd.trace_denoise_albedo = true;
            prd.trace_denoise_normal = true;
        }

        // Primary Ray
        unsigned char background_trace = 0;
        prd.alphaHit = false;

        traceRadiance(params.handle, ray_origin, ray_direction, _tmin_, prd.maxDistance, &prd, _mask_);
        float3 m = prd.mask_value;
        mask_value = mask_value + m;

        auto primary_hit_type = prd.hit_type;
        background_trace = primary_hit_type;

        if(primary_hit_type > 0) {
            result_d = prd.radiance_d * prd.attenuation2;
            result_s = prd.radiance_s * prd.attenuation2;
            result_t = prd.radiance_t * prd.attenuation2;
        }

        tmp_albedo = prd.tmp_albedo;
        tmp_normal = prd.tmp_normal;

        prd.trace_denoise_albedo = false;
        prd.trace_denoise_normal = false;

        aov[0] = result_b;
        aov[1] = result_d;
        aov[2] = result_s;
        aov[3] = result_t;

        for(;;)
        {
            _tmin_ = prd._tmin_;
            _mask_ = prd._mask_;

            prd._tmin_ = 0;
            prd._mask_ = EverythingMask;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            if(prd.countEmitted==false || prd.depth>0) {
                auto temp_radiance = prd.radiance * prd.attenuation2;

                float upperBound = prd.fromDiff?10.0f:1000.0f;
                float3 clampped = clamp(vec3(temp_radiance), vec3(0), vec3(40));

                result += prd.depth>1?clampped:temp_radiance;

                if(primary_hit_type > 0 && ( prd.depth>1 || (prd.depth==1 && prd.hit_type == 0) )) {
                    aov[primary_hit_type] += prd.depth>1?clampped:temp_radiance;
                }
            }
            prd.radiance = make_float3(0);
            prd.emission = make_float3(0);

            if(prd.countEmitted==true && prd.depth>0){
                prd.done = true;
            }

            if( prd.done || params.simpleRender==true || prd.adepth>64){
                break;
            }

            //if(prd.depth>prd.max_depth) {
            float RRprob = max(max(prd.attenuation.x, prd.attenuation.y), prd.attenuation.z);
            if(rnd(prd.seed) > RRprob || prd.depth > prd.max_depth*2) {
                prd.done=true;
            } else {
                prd.attenuation = prd.attenuation / RRprob;
            }
            //}
            if(prd.countEmitted == true)
                prd.passed = true;


            prd.radiance_d = make_float3(0);
            prd.radiance_s = make_float3(0);
            prd.radiance_t = make_float3(0);
            prd.alphaHit = false;
            prd._tmin_ = 0;
            traceRadiance(params.handle, ray_origin, ray_direction, _tmin_, prd.maxDistance, &prd, _mask_);

            if(prd.hit_type>0 && primary_hit_type==0)
            {
              primary_hit_type = prd.hit_type;
              aov[primary_hit_type] += (prd.hit_type==1?prd.radiance_d:(prd.hit_type==2?prd.radiance_s:prd.radiance_t))*prd.attenuation2;
            }
            background_trace += prd.hit_type>0?1:0;

        }
        
        seed = prd.seed;

        if (!(background_trace == 0)) {
            result_b += make_float3(1);
        }
    }
    while( --i );
    aperture      = aperture < 0.0001 ? params.physical_camera_aperture: aperture;
    float shutter_speed = params.physical_camera_shutter_speed;
    float iso           = params.physical_camera_iso;
    float aces          = params.physical_camera_aces;
    float exposure      = params.physical_camera_exposure;
    float midGray       = 0.18f;
    auto samples_per_launch = static_cast<float>( params.samples_per_launch );

    vec3         accum_color    = PhysicalCamera(vec3(result), aperture, shutter_speed, iso, midGray, exposure, aces) / samples_per_launch;
    vec3         accum_color_d  = PhysicalCamera(vec3(aov[1]), aperture, shutter_speed, iso, midGray, exposure, aces) / samples_per_launch;
    vec3         accum_color_s  = PhysicalCamera(vec3(aov[2]), aperture, shutter_speed, iso, midGray, exposure, aces) / samples_per_launch;
    vec3         accum_color_t  = PhysicalCamera(vec3(aov[3]), aperture, shutter_speed, iso, midGray, exposure, aces) / samples_per_launch;
    float3         accum_color_b  = result_b / samples_per_launch;
    float3         accum_mask     = mask_value / samples_per_launch;
    
    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        const float3 accum_color_prev_d = make_float3( params.accum_buffer_D[ image_index ]);
        const float3 accum_color_prev_s = make_float3( params.accum_buffer_S[ image_index ]);
        const float3 accum_color_prev_t = make_float3( params.accum_buffer_T[ image_index ]);
        const float3 accum_color_prev_b = make_float3( params.accum_buffer_B[ image_index ]);
        const float3 accum_mask_prev    = params.frame_buffer_M[ image_index ];
        accum_color   = mix( vec3(accum_color_prev), accum_color, a );
        accum_color_d = mix( vec3(accum_color_prev_d), accum_color_d, a );
        accum_color_s = mix( vec3(accum_color_prev_s), accum_color_s, a );
        accum_color_t = mix( vec3(accum_color_prev_t), accum_color_t, a );
        accum_color_b = lerp( accum_color_prev_b, accum_color_b, a );
        accum_mask    = lerp( accum_mask_prev, accum_mask, a);

        if (params.denoise) {

            const float3 accum_albedo_prev = params.albedo_buffer[ image_index ];
            tmp_albedo = lerp(accum_albedo_prev, tmp_albedo, a);

            const float3 accum_normal_prev = params.normal_buffer[ image_index ];
            tmp_normal = lerp(accum_normal_prev, tmp_normal, a);
        }
    }

    params.accum_buffer[ image_index ] = make_float4( accum_color.x, accum_color.y, accum_color.z, 1.0f);
    params.accum_buffer_D[ image_index ] = make_float4( accum_color_d.x,accum_color_d.y,accum_color_d.z, 1.0f);
    params.accum_buffer_S[ image_index ] = make_float4( accum_color_s.x,accum_color_s.y, accum_color_s.z, 1.0f);
    params.accum_buffer_T[ image_index ] = make_float4( accum_color_t.x,accum_color_t.y,accum_color_t.z, 1.0f);
    params.accum_buffer_B[ image_index ] = make_float4( accum_color_b, 1.0f);


    vec3 rgb_mapped = PhysicalCamera(vec3(accum_color), aperture, shutter_speed, iso, midGray, false, false);
    vec3 d_mapped = PhysicalCamera(vec3(accum_color_d), aperture, shutter_speed, iso, midGray, false, false);
    vec3 s_mapped = PhysicalCamera(vec3(accum_color_s), aperture, shutter_speed, iso, midGray, false, false);
    vec3 t_mapped = PhysicalCamera(vec3(accum_color_t), aperture, shutter_speed, iso, midGray, false, false);


    float3 out_color = rgb_mapped;
    float3 out_color_d = d_mapped;
    float3 out_color_s = s_mapped;
    float3 out_color_t = t_mapped;
    float3 out_color_b = accum_color_b;
    params.frame_buffer[ image_index ] = make_color ( out_color );
    params.frame_buffer_C[ image_index ] = out_color;
    params.frame_buffer_D[ image_index ] = out_color_d;
    params.frame_buffer_S[ image_index ] = out_color_s;
    params.frame_buffer_T[ image_index ] = out_color_t;
    params.frame_buffer_B[ image_index ] = accum_color_b;
    params.frame_buffer_M[ image_index ] = accum_mask;

    if (params.denoise) {
        params.albedo_buffer[ image_index ] = tmp_albedo;
        params.normal_buffer[ image_index ] = tmp_normal;
    }
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
    
    if(prd->medium != DisneyBSDF::PhaseFunctions::isotropic){
        float upperBound = 100.0f;
        float envPdf = 0.0f;
        vec3 skysample =
            envSky(
            normalize(prd->direction),
            sunLightDir,
            make_float3(0., 0., 1.),
            40, // be careful
            .45,
            15.,
            1.030725f * 0.3f,
            params.elapsedTime,
            envPdf,
            upperBound,
            0.0

        );

        float misWeight = BRDFBasics::PowerHeuristic(prd->samplePdf,envPdf);

        misWeight = misWeight>0.0f?misWeight:0.0f;
        misWeight = envPdf>0.0f?misWeight:1.0f;
        misWeight = prd->depth>=1?misWeight:1.0f;
        misWeight = prd->samplePdf>0.0f?misWeight:1.0f;
        
        prd->radiance = misWeight * skysample;

        if (params.show_background == false) {
            prd->radiance = prd->depth>=1?prd->radiance:make_float3(0,0,0);
        }

        prd->done      = true;
        prd->hit_type  = 0;
        return;
    }

    vec3 sigma_t, ss_alpha;
    //vec3 sigma_t, ss_alpha;
    prd->readMat(sigma_t, ss_alpha);


    vec3 transmittance;
    if (ss_alpha.x < 0.0f) { // is inside Glass
        transmittance = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax());
    } else {
        transmittance = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax(), false);
    }

    prd->attenuation *= transmittance;//DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->attenuation2 *= transmittance;//DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->origin += prd->direction * optixGetRayTmax();
    prd->direction = DisneyBSDF::SampleScatterDirection(prd->seed);

    vec3 channelPDF = vec3(1.0f/3.0f);
    prd->channelPDF = channelPDF;
    if (ss_alpha.x < 0.0f) { // is inside Glass
        prd->maxDistance = DisneyBSDF::SampleDistance(prd->seed, prd->scatterDistance);
    } else
    {
        prd->maxDistance =
            DisneyBSDF::SampleDistance2(prd->seed, vec3(prd->attenuation) * ss_alpha, sigma_t, channelPDF);
        prd->channelPDF = channelPDF;
    }

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
