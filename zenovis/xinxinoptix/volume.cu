#include "Light.h"
#include "volume.h"

#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "math_constants.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/SampleFromVoxels.h>

using DataTypeNVDB0 = nanovdb::Fp32;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;

extern "C" __global__ void __intersection__volume()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>( optixGetSbtDataPointer() );
    const auto* grid = reinterpret_cast<const GridTypeNVDB0*>( sbt_data->vdb_grids[0] );
    if ( grid == nullptr) { return; }

    const float3 ray_orig = optixGetWorldRayOrigin() + params.cam.eye;
    const float3 ray_dir  = optixGetWorldRayDirection();

    auto dbox = grid->worldBBox(); //grid->indexBBox();
    float t0 = optixGetRayTmin();
    float t1 = _FLT_MAX_; //optixGetRayTmax();

    auto iray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_orig ),
                                     reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ), t0, t1 );
    // auto fbox = nanovdb::BBox<nanovdb::Vec3f>(nanovdb::Vec3f(dbox.min()), nanovdb::Vec3f(dbox.max()));

    auto flags = optixGetRayFlags();
    auto anyhit = flags & OPTIX_RAY_FLAG_ENFORCE_ANYHIT;

    if( iray.intersects( dbox, t0, t1 )) // t0 >= 0
    {
        // report the entry-point as hit-point
        //auto kind = optixGetHitKind();
        t0 = fmaxf(t0, optixGetRayTmin());

        if (anyhit) {
            ShadowPRD *prd = getPRD<ShadowPRD>();

            prd->vol.vol_t0 = t0;
            prd->vol.origin_inside = (t0 == 0);
            prd->vol.vol_t1 = t1; //min(optixGetRayTmax(), t1);
            prd->vol.surface_inside = (optixGetRayTmax() < t1);
        } else {

            RadiancePRD* prd = getPRD();
            prd->vol.vol_t0 = t0;
            prd->vol.origin_inside = (t0 == 0);

            prd->vol.vol_t1 = t1; //min(optixGetRayTmax(), t1);
            prd->vol.surface_inside = (optixGetRayTmax() < t1); // In case triangles were visited before volume
        }

        if (optixGetRayTmax() > 0) {
            optixReportIntersection(t0, 0);
        }
    } 
}

extern "C" __global__ void __closesthit__radiance_volume()
{
    RadiancePRD* prd = getPRD();
    
    prd->countEmitted = false;
    prd->radiance = vec3(0);

    prd->_tmin_ = 0;
    prd->_mask_ = EverythingMask;

    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

        float t0 = prd->vol.vol_t0; // world space
        float t1 = prd->vol.vol_t1; // world space

    RadiancePRD testPRD {};
    testPRD.isSS = false;
    testPRD.maxDistance = _FLT_MAX_;
    testPRD.test_distance = true;
    
    uint16_t _mask_ = EverythingMask ^ VolumeMatMask;

    traceRadiance(params.handle, ray_orig,ray_dir, 0, _FLT_MAX_, &testPRD, _mask_);

    if(testPRD.maxDistance < t1)
    {
        t1 = testPRD.maxDistance;
        prd->vol.surface_inside = true;
    }

    const float t_max = fmax(0.f, t1 - t0); // world space
    float t_ele = 0;

    float3 new_orig = ray_orig; 
    float3 emitting = make_float3(0.0);
    float3 scattering = make_float3(1.0);
   
    float sigma_t = sbt_data->vol_extinction;
    float v_density = 0.0;

    VolumeOut vol_out;
    auto new_dir = ray_dir;

    auto level = sbt_data->vol_depth;
    auto step_scale = 1.0f/sigma_t;

    while(--level > 0) {
        auto prob = prd->rndf();
        t_ele -= logf(1.0f-prob) * step_scale;

        if (t_ele >= t_max) {

            if (prd->vol.surface_inside) { // Hit other material

                prd->_mask_ = _mask_;
                prd->_tmin_ = 0;

                new_orig = ray_orig;

            } else { // Volume edge

                prd->_mask_ = EverythingMask;
                prd->_tmin_ = 1e-5f;

                new_orig = ray_orig + t1 * ray_dir;
                new_orig = rtgems::offset_ray(new_orig, ray_dir);
            }

            v_density = 0;
            break;
        } // over shoot, outside of volume

        new_orig = ray_orig + (t0+t_ele) * ray_dir;

        VolumeIn vol_in { new_orig+params.cam.eye, sigma_t, &prd->seed, reinterpret_cast<unsigned long long>(sbt_data) };

        vol_out = optixDirectCall<VolumeOut, const float4*, const VolumeIn&>( sbt_data->dc_index, sbt_data->uniforms, vol_in);
        v_density = vol_out.density;
        emitting += vol_out.emission;

        step_scale = fminf(step_scale, vol_out.step_scale) ;

        if (prd->rndf() > v_density) { // null scattering
            v_density = 0.0f; continue;
        }

        pbrt::HenyeyGreenstein hg (vol_out.anisotropy);
        float2 uu = { prd->rndf(), prd->rndf() };
        auto pdf = hg.sample(-ray_dir, new_dir, uu);              
        //auto relative_prob = prob * (CUDART_PI_F * 4);
        prd->samplePdf = pdf;

        new_dir = normalize(new_dir);
        scattering = vol_out.albedo;

        if (prd->trace_denoise_normal) {
            prd->tmp_normal = normalize(-ray_dir + new_dir);
        }
        if(prd->trace_denoise_albedo) {
            prd->tmp_albedo = vol_out.albedo;
        }

        break;
    }

    prd->updateAttenuation(scattering);

    prd->origin = new_orig;
    prd->direction = new_dir;

    prd->emission = emitting;
    prd->geometryNormal = {}; //(new_dir + -ray_dir) / 2.0f;

    if (v_density == 0) {
        //prd->depth += 0;
        prd->radiance += prd->emission;
        return;
    }

    scattering = vol_out.albedo;

    ShadowPRD shadowPRD {};
    shadowPRD.seed = prd->seed;
    shadowPRD.origin = new_orig; //camera sapce
    shadowPRD.attanuation = vec3(1.0f);
    
    auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {

        pbrt::HenyeyGreenstein hg(vol_out.anisotropy);
        thisPDF = hg.p(_wo_, _wi_);
        return scattering * thisPDF;
    };

    DirectLighting<true>(prd, shadowPRD, new_orig+params.cam.eye, ray_dir, evalBxDF);
    
    prd->depth += 1;
    prd->radiance += prd->emission;
    
    return;
}

extern "C" __global__ void __anyhit__occlusion_volume()
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    ShadowPRD* prd = getPRD<ShadowPRD>();
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const float t0 = prd->vol.vol_t0;
    const float t1 = prd->vol.vol_t1;

    const float t_max = t1 - t0; // world space
          float t_ele = 0;

    float3 test_point = ray_orig; 
    float3 transmittance = make_float3(1.0f);

    float hgp = 1.0f;
    pbrt::HenyeyGreenstein hg(9.0f);
    
    const float sigma_t = sbt_data->vol_extinction;

    auto level = sbt_data->vol_depth;
    while(--level > 0) {

        auto prob = prd->rndf();
        t_ele -= log(1.0f-prob) / (sigma_t);

        test_point = ray_orig + (t0+t_ele) * ray_dir;

        if (t_ele >= t_max) {
            break;
        } // over shoot, outside of volume

        VolumeIn vol_in { test_point+params.cam.eye, sigma_t, &prd->seed, reinterpret_cast<unsigned long long>(sbt_data) };
        VolumeOut vol_out = optixDirectCall<VolumeOut, const float4*, const VolumeIn&>( sbt_data->dc_index, sbt_data->uniforms, vol_in );

        const auto v_density = vol_out.density;

        auto prob_scatter = clamp(v_density, 0.0f, 1.0f);
        auto prob_nulling = 1.0f - prob_scatter;

        if (vol_out.anisotropy != hg.g ) {
            hg = pbrt::HenyeyGreenstein(vol_out.anisotropy);
            hgp = hg.p(-ray_dir, ray_dir);
        }

        float prob_continue = hgp * prob_scatter;
        prob_continue = clamp(prob_continue, 0.0, prob_scatter);

        auto tr = transmittance * prob_nulling;
        tr += transmittance * prob_continue * vol_out.albedo;
        
        transmittance = clamp(tr, 0.0, 1.0f);

        auto avg = dot(transmittance, make_float3(1.0f/3.0f));
        if (avg < 0.1f) {
            float q = fmax(0.05f, 1 - avg);
            if (prd->rndf() < q) { 
                transmittance = vec3(0);
                break; 
            } else {
                transmittance /= 1-q;
            }
        }
    }

    prd->attanuation *= transmittance;
    optixIgnoreIntersection();
    //prd->origin = ray_orig;
    //prd->direction = ray_dir;
    return;
}