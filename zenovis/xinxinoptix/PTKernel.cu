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
#include <volume.h>
#include <Light.h>
#include <HUD.h>

#ifndef __CUDACC_RTC__
#define __AOV__ 1
#define DENOISE 1
#endif

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
__inline__ __device__ bool isBadVector(const float3 & vector) {

    bool bad = !isfinite(vector.x) || !isfinite(vector.y) || !isfinite(vector.z);
    return bad? true : lengthSquared(vector) == 0.0f;
}

void homoVolumeLight(const RadiancePRD& prd, float _tmax_, float3 ray_origin, float3 ray_dir, float3& result, float3& _attenuation) {
    
    const auto& vol = prd.vol;
    // if (vol.homo_t1 <= vol.homo_t0) return;
    // if (_tmax_ <= vol.homo_t0) return;
    
    float tmax = fminf(_tmax_, vol.homo_t1) - vol.homo_t0;

    VolumeOut homo_out;
    optixDirectCall<void, void*, VolumeOut&>( prd.vol.homo_matid, nullptr, homo_out);

    const auto& extinction = homo_out.extinction;
    auto seed = prd.seed;

    auto total_transmittance = expf(-extinction * tmax);
    float x = 1.0 - rnd(seed) * (1.0 - average(total_transmittance));
    float dt = -logf(x) / average(extinction);
    auto cdf = 1.0f - total_transmittance;
    auto pdf = expf(-extinction * dt) * extinction;
    auto weight = cdf / pdf;

    const auto& e = extinction;
    weight.x = e.x>1e-4? weight.x : (tmax * expf(-e.x * tmax))/(-dt*expf(-e.x*dt)*e.x + expf(-e.x*dt));
    weight.y = e.y>1e-4? weight.y : (tmax * expf(-e.y * tmax))/(-dt*expf(-e.y*dt)*e.y + expf(-e.y*dt));
    weight.z = e.z>1e-4? weight.z : (tmax * expf(-e.z * tmax))/(-dt*expf(-e.z*dt)*e.z + expf(-e.z*dt));

    weight *= extinction;

    let new_orig = ray_origin + (prd.vol.homo_t0 + dt) * ray_dir;

    ShadowPRD shadowPRD {};
    shadowPRD.seed = seed ^ 0x9e3779b9u;
    
    shadowPRD.depth = prd.depth;
    shadowPRD.origin = new_orig;
    shadowPRD.attanuation = vec3(1.0f);

    auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {
        pbrt::HenyeyGreenstein hg(homo_out.anisotropy);
        thisPDF = hg.p(_wo_, _wi_);
        return homo_out.albedo * thisPDF * homo_out.albedoAmp;
    };

    DirectLighting<true>(shadowPRD, new_orig+params.cam.eye, ray_dir, evalBxDF);
    shadowPRD.radiance *= weight;

    result += _attenuation * shadowPRD.radiance * expf(-extinction * dt);
    _attenuation *= expf(-extinction * tmax);
};

extern "C" __global__ void __raygen__rg()
{
    const auto w = params.width;
    const auto h = params.height;

    uint3 idx = optixGetLaunchIndex();
    if(idx.x>w || idx.y>h)
        return;

    const unsigned int image_index  = idx.y * w + idx.x;
    const int    subframe_index = params.subframe_index;
    const CameraInfo cam = params.cam;

    int seedy = idx.y/4, seedx = idx.x/8;
    int sid = (idx.y%4) * 8 + idx.x%8;

    unsigned int seed0;
    unsigned int seed;
    unsigned int eventseed;
    unsigned int seed1;

    //seed0 is fixed for a pixel, at subframe_index = 0:
    seed0 = tea<4>( idx.y * w + idx.x, 0) + params.outside_random_number;
    seed0 = pcg_hash(seed0);
    rnd(seed0);

    //seed changes every subframe
    seed = tea<4>( idx.y * w + idx.x, subframe_index) + params.outside_random_number;
    seed = pcg_hash(seed);
    rnd(seed);

    //eventseed, which is used for sobol random number per pixel
    //shall be simply seed0 + subframe_index, because it shall come at
    //squence!!
    eventseed = seed0 + subframe_index;

    //vdcseed, which is used for vdc sequence permulation, shall
    //stay fixed for subframes of a pixel!
    unsigned int vdcseed = seed0;

    //vdc offset, which is used to draw elements from the vdc sequence
    //shall increase exactly as the subframe_index!
    seed1 = seed0 + subframe_index;

    float focalPlaneDistance = cam.focal_distance>0.01f? cam.focal_distance: 0.01f;
    float aperture = clamp(cam.aperture,0.0f,100.0f);
    float physical_aperture = 0.0f;
    if(aperture < 0.05f || aperture > 24.0f){
        physical_aperture = 0.0f;
    }else{
        physical_aperture = cam.focal_length / aperture;
    }
    
    float3 result = make_float3( 0.0f );

#if __AOV__
    float3 aov[4] {};
    float3& result_d = aov[1];
    float3& result_s = aov[2];
    float3& result_t = aov[3];
    float&  result_b = aov[0].x;
#endif

    int i = params.samples_per_launch;
    
#if DENOISE
    float3 tmp_albedo{};
    float3 tmp_normal{};
#endif 

    unsigned int sobolseed = subframe_index;
    float3 mask_value = make_float3( 0.0f );

    do{
        // The center of each pixel is at fraction (0.5,0.5)
        float2 subpixel_jitter = mmd(eventseed);
//        subpixel_jitter.x = pcg_rng(seed);
//        subpixel_jitter.y = pcg_rng(seed);

        float2 d = 2.0f * make_float2(
            ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
            ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
            ) - 1.0f;

        float2 r01 = {rnd(seed), rnd(seed)};

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
        float3 focal_plane_center = make_float3(cam.horizontal_shift*cam.width, cam.vertical_shift*cam.height, cam.focal_length);
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
        if (params.physical_camera_panorama_camera) {
            ray_origin    = make_float3(0.0f, 0.0f, 0.0f);
            float phi = (float(idx.x) + subpixel_jitter.x) / float(w) * 2.0f * M_PIf;
            mat3 camera_transform = mat3(
                    cam.right.x, cam.up.x, -cam.front.x,
                    cam.right.y, cam.up.y, -cam.front.y,
                    cam.right.z, cam.up.z, -cam.front.z
            );
            if (params.physical_camera_panorama_vr180) {
                int idxx = idx.x >= w/2? idx.x - w/2 : idx.x;
                phi = ((float(idxx) + subpixel_jitter.x) / float(w / 2) + 0.5f) * M_PIf;
                if (idx.x < w / 2) {
                    ray_origin = camera_transform * make_float3(-params.physical_camera_pupillary_distance / 2.0f, 0.0f, 0.0f);
                }
                else {
                    ray_origin = camera_transform * make_float3(params.physical_camera_pupillary_distance / 2.0f, 0.0f, 0.0f);
                }
            }
            float theta = (float(idx.y) + subpixel_jitter.y) / float(h) * M_PIf;
            float y = -cosf(theta);
            float z = sinf(theta) * cosf(phi);
            float x = sinf(theta) * sinf(-phi);

            ray_direction = camera_transform * make_float3(x, y, z);
        }

        RadiancePRD prd;
        prd.print_info = params.click_dirty && params.click_coord.x==idx.x && params.click_coord.y==idx.y;
        prd.vdcseed = vdcseed;
        prd.offset = seed1;
        prd.offset2 = seed1;
        prd.offset3 = seed1;
        prd.pixel_area   = cam.height/(float)(h)/(cam.focal_length);

        prd.emission     = make_float3(0.f);
        prd.radiance     = make_float3(0.f);
        prd.attenuation  = make_float3(1.f);
        prd.countEmitted = true;
        prd.done         = false;
        prd.seed         = seed;
        prd.eventseed    = eventseed;
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
        prd.ss_alpha_queue[0] = half3(-1.0f);
        prd.minSpecRough = 0.01;
        prd.samplePdf = 1.0f;
        prd.hit_type = 0;
        prd.max_depth = 4;
        auto _tmin_ = prd._tmin_;
        auto _mask_ = prd._mask_;
    #if __AOV__ 
        prd.__aov__ = true;
    #endif
    #if DENOISE 
        prd.denoise = true;
    #endif
        rnd(prd.seed);
        vdcrnd(prd.offset, prd.vdcseed);
        vdcrnd(prd.offset, prd.vdcseed);
        vdcrnd(prd.offset, prd.vdcseed);

        // Primary Ray
        auto _attenuation = prd.attenuation;
        do {
            prd.alphaHit = false;
            traceRadiance(params.handle, ray_origin, ray_direction, prd._tmin_, prd.maxDistance, &prd, _mask_);
        } while (prd.alphaHit); // skip alpha

        if ( params.click_dirty && params.click_coord.x==idx.x && params.click_coord.y==idx.y )
        {
            float3 click_pos {0,0,0};
            uint4 record {0,0,0,0};

            if (prd._tmax_ < FLT_MAX) {
                click_pos = ray_origin + ray_direction * prd._tmax_;
                record = prd.record;
            }
            *params.pick_buffer = PickInfo { click_pos, record };
        }
        if(params.pause) return;
        
        prd._tmin_ = 0;
        //prd._tmax_ = FLT_MAX;
        prd.maxDistance = FLT_MAX;
        
        float3 m = prd.mask_value;
        mask_value = mask_value + m;

    #if __AOV__
        const auto primary_hit_type = prd.hit_type;
        result_b += primary_hit_type? 1:0;
        if(primary_hit_type > 0) {
            result_d = prd.aov[0] * _attenuation;
            result_s = prd.aov[1] * _attenuation;
            result_t = prd.aov[2] * _attenuation;
        }
    #endif

    #if DENOISE
        tmp_albedo = prd.tmp_albedo;
        tmp_normal = prd.tmp_normal;
        prd.denoise = false;
    #endif

        for(;;)
        {
            _tmin_ = prd._tmin_;
            _mask_ = prd._mask_;
            if (prd.vol.homo_t1 > prd.vol.homo_t0 && prd._tmax_ > prd.vol.homo_t0)
                homoVolumeLight(prd, prd._tmax_, ray_origin, ray_direction, result, _attenuation);
            
            prd.vol = {};
            prd._tmin_ = _tmin_;
            prd._tmax_ = FLT_MAX;
            prd._mask_ = EverythingMask;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            if(prd.countEmitted==false || prd.depth>0) {
                auto temp_radiance = prd.radiance * _attenuation;

                float upperBound = prd.fromDiff?10.0f:1000.0f;
                float3 clampped = clamp(vec3(temp_radiance), vec3(0), vec3(10.0f));

                result += prd.depth>1?clampped:temp_radiance;
            #if __AOV__
                if(primary_hit_type > 0 && ( prd.depth>1 || (prd.depth==1 && prd.hit_type == 0) )) {
                    aov[primary_hit_type] += prd.depth>1?clampped:temp_radiance;
                }
            #endif
            }
            prd.radiance = make_float3(0);
            prd.emission = make_float3(0);

            if(prd.countEmitted==true && prd.depth>0){
                prd.done = true;
            }

            if( prd.done || prd.depth>prd.max_depth){
                break;
            }

            if(prd.depth > 1){
                float RRprob = max(max(prd.attenuation.x, prd.attenuation.y), prd.attenuation.z);
                RRprob = min(RRprob, 0.99f);
                if(rnd(prd.seed) > RRprob) {
                    break;
                } else {
                    prd.attenuation = prd.attenuation / RRprob;
                }
            }


            if(prd.diffDepth > 1)
                _mask_ &= ~VolumeMaskAnalytics;
            prd._tmin_ = _tmin_;
            do {
                _attenuation = prd.attenuation;
                prd.alphaHit = false;
                traceRadiance(params.handle, ray_origin, ray_direction, prd._tmin_, prd.maxDistance, &prd, _mask_);
            }while(prd.alphaHit);
        }
        seed = prd.seed;
//        seed1 = prd.offset;
//        eventseed = prd.eventseed;
    }
    while( --i );
    aperture      = aperture < 0.0001 ? params.physical_camera_aperture: aperture;
    float shutter_speed = params.physical_camera_shutter_speed;
    float iso           = params.physical_camera_iso;
    float midGray       = 0.18f;

    bool need_manual_exposure = params.physical_camera_exposure;
    bool need_tone_mapping = params.physical_camera_aces;

    auto samples_per_launch = static_cast<float>( params.samples_per_launch );

    const auto tmp = 1.0f / params.samples_per_launch;
    auto& accum_color    = result; accum_color *= tmp;
#if __AOV__
    auto& accum_color_d  = aov[1]; accum_color_d *= tmp;
    auto& accum_color_s  = aov[2]; accum_color_s *= tmp;
    auto& accum_color_t  = aov[3]; accum_color_t *= tmp;

    auto accum_color_b  = result_b * tmp;
    auto accum_mask     = mask_value * tmp;
#endif

    if (need_manual_exposure) {
        auto manual_exposure = [&](float3& color){
            color = PhysicalCamera(color, aperture, shutter_speed, iso, midGray, true, false);
        };
        manual_exposure(accum_color);
    #if __AOV__ 
        manual_exposure(accum_color_d);
        manual_exposure(accum_color_s);
        manual_exposure(accum_color_t);
    #endif
    }

    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev   = params.accum_buffer[ image_index ];
        accum_color = mix( accum_color_prev, accum_color, a );
    #if __AOV__
        const float3 accum_color_prev_d = params.accum_buffer_D[ image_index ];
        const float3 accum_color_prev_s = params.accum_buffer_S[ image_index ];
        const float3 accum_color_prev_t = params.accum_buffer_T[ image_index ];
        const float accum_color_prev_b  = __half2float(*(__half*)&params.accum_buffer_B[image_index]);
        const float3 accum_mask_prev    = half3_to_float3(params.frame_buffer_M[ image_index ]);

        accum_color_d = mix( vec3(accum_color_prev_d), accum_color_d, a );
        accum_color_s = mix( vec3(accum_color_prev_s), accum_color_s, a );
        accum_color_t = mix( vec3(accum_color_prev_t), accum_color_t, a );
        accum_color_b = mix( accum_color_prev_b, accum_color_b, a );
        accum_mask    = lerp( accum_mask_prev, accum_mask, a);
    #endif

        #if DENOISE
            const float3 accum_albedo_prev = params.albedo_buffer[ image_index ];
            tmp_albedo = lerp(accum_albedo_prev, tmp_albedo, a);
            const float3 accum_normal_prev = params.normal_buffer[ image_index ];
            tmp_normal = lerp(accum_normal_prev, tmp_normal, a);

            params.albedo_buffer[ image_index ] = tmp_albedo;
            params.normal_buffer[ image_index ] = tmp_normal;
        #endif
    }

    params.accum_buffer[ image_index ] = accum_color;

    #if __AOV__
        params.accum_buffer_D[ image_index ] = accum_color_d;
        params.accum_buffer_S[ image_index ] = accum_color_s;
        params.accum_buffer_T[ image_index ] = accum_color_t;
        auto h3 = float3_to_half3(accum_mask);
        params.frame_buffer_M[ image_index ] = reinterpret_cast<ushort3&>(h3);
        auto accum_buffer_B = reinterpret_cast<__half*>(params.accum_buffer_B);
        accum_buffer_B[image_index] = __float2half(accum_color_b);
    #endif

    auto uv = float2{idx.x+0.5f, idx.y+0.5f};
    auto dither = InterleavedGradientNoise(uv);

    dither = (dither-0.5f);
    if (need_tone_mapping) {
        accum_color = ACESFilm(accum_color);
    }
    auto& pixel = params.frame_buffer[image_index];
    pixel = makeSRGB( accum_color, 2.2f, dither);

    if (params.frame_time > 0) {
        drawHUD((uchar3*)&pixel, params.frame_time, uv/make_float2(w/16, 20));
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
    prd->countEmitted = false;
    prd->radiance *= 0;
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

        envPdf *= params.skyLightProbablity();

        float misWeight = BRDFBasics::PowerHeuristic(prd->samplePdf,envPdf, 1.0f);

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
        transmittance = DisneyBSDF::Transmission(sigma_t, optixGetRayTmax() - prd->_tmin_);
    } else {
        transmittance = DisneyBSDF::Transmission2(sigma_t * ss_alpha, sigma_t, prd->channelPDF, optixGetRayTmax() - prd->_tmin_, false);
    }

    prd->attenuation *= transmittance;//DisneyBSDF::Transmission(prd->extinction,optixGetRayTmax());
    prd->attenuation2*= transmittance;
    prd->origin += prd->direction * ( optixGetRayTmax() - prd->_tmin_);
    prd->_tmin_ = 0.0f;
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
    auto flags = OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT; 
    if (optixGetRayFlags() == flags){
        ShadowPRD* prd = getPRD<ShadowPRD>();
        prd->attanuation = vec3(1.0f);
    }
}

