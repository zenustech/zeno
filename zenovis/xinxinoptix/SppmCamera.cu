#include <cuda/helpers.h>
#include "CameraRay.h"
#include "SppmDevice.h"

extern "C" { __constant__ Params params; }

extern "C" __global__ void __raygen__sppm_camera() {
    const auto index = optixGetLaunchIndex();
    const uint32_t i = index.y * params.width + index.x;
    auto& pixel = params.sppm.pixels[i];
    pixel.visible = pixel.m = 0u;
    pixel.phi = make_float3(0.0f);
    if (params.sppm.fgEnabled) {
        // Initialize storage once; validity controls all subsequent reads.
        params.sppm.fgCandidates[i].point.valid = 0u;
        params.sppm.fgInitial[i].source.valid = 0u;
    }

    uint32_t primarySeed=pcg_hash(tea<4>(i,params.sppm.iteration)+params.outside_random_number);
    const uint32_t vdcseed=pcg_hash(tea<4>(i,0u)+params.outside_random_number);
    uint32_t eventseed=vdcseed+params.sppm.iteration;
    float3 origin,direction;
    const float2 jitter=mmd(eventseed);
    zenoCameraRay(index,jitter,primarySeed,origin,direction);
    const float3 primaryOrigin = origin, primaryDirection = direction;
    float3 direct = make_float3(0.0f);
    float3 albedo = {}, normal = {};
    uint32_t cameraSamples = 1u;
    // Replica zero is an independent pilot and the photon visible-point path.
    // Its radiance selects effort but is excluded from the displayed average.
    // Reusing prior pilot statistics would correlate the budget with the next
    // point in that same scrambled Sobol sequence.
    for (uint32_t replica = 0u; replica <= cameraSamples; ++replica) {
        // One live transport record. Retaining a second complete PRD snapshot
        // across every trace caused large local/continuation stack spills.
        RadiancePRD prd={};
        SppmCameraShadow shadow;
        prd.sppmCameraShadow=&shadow;
        prd.sppmMode=1u; prd.sppmPixel=i;
        prd.seed=primarySeed; prd.vdcseed=vdcseed;
        prd.offset=prd.offset2=prd.offset3=vdcseed+params.sppm.iteration;
        prd.eventseed=eventseed;
        prd.ss_alpha_queue[0]=half3(-1.0f);
        prd.attenuation=make_float3(1.0f);
        prd.max_depth=uint8_t(params.max_bounce);
        prd.maxDistance=FLT_MAX;
        prd.denoise=params.denoise;
        prd.pixel_area=params.cam.height/float(params.height)/params.cam.focal_length;
        prd.sppmCameraSample = replica;
        if (replica != 0u) prd.denoise = false;
        origin = primaryOrigin;
        direction = primaryDirection;
        if (replica != 0u) {
            prd.seed = pcg_hash(primarySeed ^ tea<4>(replica, 0x193ab45du));
            const float2 subpixel = make_float2(DisneyBSDF::sppmCameraSample(prd, 6u),
                                               DisneyBSDF::sppmCameraSample(prd, 7u));
            zenoCameraRay(index, subpixel, prd.seed, origin, direction);
        }
        float3 pathDirect = make_float3(0.0f);
        for (uint32_t depth = 0; depth < params.max_bounce && !prd.done; ++depth) {
            prd.origin = origin;
            prd.direction = direction;
            prd.radiance = make_float3(0.0f);
            const float3 beta = prd.attenuation;
            const bool beforeFinalGather = prd.fgStage == 0u;
            float tmin = 0.0f;
            do {
                prd.alphaHit = false;
                shadow.pending=false;
                prd._tmin_ = 0.0f;
                traceRadiance(params.handle, origin, direction, tmin, FLT_MAX,
                              &prd, EverythingMask ^ VolumeMatMask);
                tmin = prd._tmin_;
            } while (prd.alphaHit && !prd.done);
            if (shadow.pending) {
                traceOcclusion(params.handle,shadow.state.spawnOrigin(shadow.direction),shadow.direction,
                    0,shadow.distance,&shadow.state,~LightMatMask&EverythingMask);
                prd.radiance+=shadow.radiance*shadow.state.attanuation;
            }
            if (beforeFinalGather) pathDirect += beta * prd.radiance;
            if (depth == 0u && replica == 0u) {
                albedo = prd.tmp_albedo;
                normal = prd.tmp_normal;
                if (params.click_dirty && params.click_coord.x == index.x && params.click_coord.y == index.y) {
                    // Match PT's no-hit sentinel. A sky miss has t=FLT_MAX;
                    // using it as a navigation pivot destroys camera/IAS bounds.
                    PickInfo pick = {};
                    if (prd._tmax_ < FLT_MAX) {
                        pick.pos = origin + direction*prd._tmax_;
                        pick.meta = prd.record;
                    }
                    *params.pick_buffer = pick;
                }
                prd.denoise = false;
            }
            if (prd.done || !sppmFinite(prd.attenuation)) break;
            if (depth >= 3u && prd.fgStage == 0u) {
                // Refraction's radiance scaling is reversed on leaving the glass.
                // It must not cause premature roulette and large surviving weights.
                const float survival = fminf(0.95f, sppmMax(prd.attenuation) * prd.sppmEtaScale);
                if (!(survival > 0.0f) || rnd(prd.seed) >= survival) break;
                prd.attenuation /= survival;
            }
            origin = prd.origin;
            direction = prd.direction;
        }
        if (replica == 0u) {
            const double value = double(pathDirect.x) * .2126 +
                double(pathDirect.y) * .7152 + double(pathDirect.z) * .0722;
            const double count = double(params.sppm.iteration) + 1.0;
            if (params.sppm.iteration == 0u) {
                pixel.cameraMean = pixel.cameraM2 = 0.0;
            }
            const double delta = value - pixel.cameraMean;
            pixel.cameraMean += delta / count;
            pixel.cameraM2 += delta * (value - pixel.cameraMean);
            const double variance = count > 1.0 ? fmax(0.0, pixel.cameraM2 / (count - 1.0)) : 0.0;
            // Bound work, not energy. Each of these paths retains its full
            // contribution, including values greater than one or ten.
            cameraSamples = uint32_t(fmin(32.0, fmax(double(prd.sppmCameraSamples), ceil(sqrt(variance)))));
            pixel.cameraSamples = cameraSamples;
        } else {
            direct += pathDirect;
        }
    }
    // Equal weight per iteration, regardless of the independent pilot's budget.
    // Photon normalization, radius and the visible-point path remain unchanged.
    direct /= float(cameraSamples);
    pixel.direct += direct;
    if (params.denoise) {
        const float a = 1.0f/(float(params.sppm.iteration)+1.0f);
        params.albedo_buffer[i] = params.sppm.iteration == 0u ? albedo : lerp(params.albedo_buffer[i], albedo, a);
        params.normal_buffer[i] = params.sppm.iteration == 0u ? normal : lerp(params.normal_buffer[i], normal, a);
    }
}
