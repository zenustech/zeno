#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

#include "Light.h"
#include "volume.h"

#include "TraceStuff.h"
#include "zxxglslvec.h"
#include "math_constants.h"

// #include <cuda_fp16.h>
// #include "nvfunctional"

using DataTypeNVDB0 = nanovdb::Fp32;
using GridTypeNVDB0 = nanovdb::NanoGrid<DataTypeNVDB0>;

__inline__ __device__ bool rayBox(const float3& ray_ori, const float3& ray_dir, const nanovdb::BBox<nanovdb::Vec3f>& box, 
                        float& t0, float& t1) {

    auto iray = nanovdb::Ray<float>( reinterpret_cast<const nanovdb::Vec3f&>( ray_ori ),
                                     reinterpret_cast<const nanovdb::Vec3f&>( ray_dir ), t0, t1 );
    return iray.intersects( box, t0, t1 );
}

extern "C" __global__ void __intersection__volume()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>( optixGetSbtDataPointer() );
    const auto* grid = reinterpret_cast<const GridTypeNVDB0*>( sbt_data->vdb_grids[0] );

    auto box = [&]() -> nanovdb::BBox<nanovdb::Vec3f> {
        if ( grid == nullptr) {
            return nanovdb::BBox<nanovdb::Vec3f>(nanovdb::Vec3f(-0.5f), nanovdb::Vec3f(0.5f));
        } else {
            auto& ibox = grid->indexBBox();
            return nanovdb::BBox<nanovdb::Vec3f>(ibox.min(), ibox.max()+nanovdb::Coord(1));
        }
    } ();

    const float3 ray_ori = optixGetObjectRayOrigin();
          float3 ray_dir = optixGetObjectRayDirection(); // not normalized

    ray_dir = normalize(ray_dir);

    float t0 = optixGetRayTmin();
    float t1 = 1e16f; //optixGetRayTmax();

    auto dirlen = length(optixGetObjectRayDirection());
    
    { // world distance to object distance 
        t0 = t0 * dirlen; 
        t1 = t1 * dirlen;
    } 

    //bool inside = box.isInside(reinterpret_cast<const nanovdb::Vec3f&>(ray_ori));
    auto hitted = rayBox( ray_ori, ray_dir, box, t0, t1 );
    if (!hitted) { return; }

    RadiancePRD *prd = getPRD<RadiancePRD>();

    //auto scale = optixTransformVectorFromObjectToWorldSpace(ray_dir);
    auto len = 1.0f / dirlen;
    // object distance to world distance 
    t0 = t0 * len;
    t1 = t1 * len;
    
    t0 = max(t0, 0.0f);
    t1 = max(t1, 0.0f);

    // report the entry-point as hit-point
    //auto kind = optixGetHitKind();
    t0 = fmaxf(t0, optixGetRayTmin());

    if (t1 <= t0) { // skip tmin
        return;
    }

    auto flags = optixGetRayFlags();
    auto anyhit = flags & OPTIX_RAY_FLAG_ENFORCE_ANYHIT;

    if (anyhit) {

        ShadowPRD *prd = getPRD<ShadowPRD>();
        prd->vol.t0 = t0;
        prd->vol.t1 = min(prd->maxDistance, t1);

    } else {

        RadiancePRD* prd = getPRD();
        prd->vol.t0 = t0;
        prd->vol.t1 = t1; //min(optixGetRayTmax(), t1);
    }

    if (optixGetRayTmax() > 0) {
        optixReportIntersection(t0, 0);
    }
}

__forceinline__ __device__ auto EvalVolume(uint32_t* seed, float* m16, float sigma_t, float3& pos) {

    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    VolumeIn vin;
    vin.pos_view = pos;
    vin.pos_world = pos + params.cam.eye;
    vin.seed = seed;
    vin.sigma_t = sigma_t;
    vin.sbt_ptr = (void*)sbt_data;
    
    vin.world2object = m16;

    return optixDirectCall<VolumeOut, const float4*, const VolumeIn&>( sbt_data->dc_index, sbt_data->uniforms, vin );
}

extern "C" __global__ void __closesthit__radiance_volume()
{
    RadiancePRD* prd = getPRD();

    prd->countEmitted = false;
    prd->radiance = vec3(0);

    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    float t0 = prd->vol.t0; // world space
    float t1 = prd->vol.t1; // world space

    RadiancePRD testPRD {};
    testPRD.done = false;
    testPRD.seed = prd->seed;
    testPRD.depth == 0;
    testPRD._tmin_ = t0;
    testPRD.maxDistance = t1;
    testPRD.test_distance = true;
    
    uint8_t _mask_ = EverythingMask ^ VolumeMatMask;

    do {
        traceRadiance(params.handle, ray_orig, ray_dir, testPRD._tmin_, testPRD.maxDistance, &testPRD, _mask_);
    } while(testPRD.test_distance && !testPRD.done);

    bool surface_inside = false;
    if(testPRD.maxDistance < t1)
    {
        t1 = testPRD.maxDistance;
        surface_inside = true;
    }

    const float t_max = fmaxf(0.f, t1 - t0); // world space

    float3 new_orig = ray_orig; 
    float3 new_dir  = ray_dir;

    auto cihouVolumeEdge = [&]() {

        if (surface_inside) { // Hit other material

            prd->_mask_ = _mask_;
            prd->_tmin_ = NextFloatDown(t1);

        } else { // Volume edge

            prd->_mask_ = EverythingMask;
            prd->_tmin_ = t1;
        }

        new_orig = ray_orig;
    };

    float m16[16] = {}; m16[15] = 1;
    optixGetWorldToObjectTransformMatrix(m16);

    if (0 == sbt_data->vol_depth) { // Homogeneous

        new_orig = ray_orig + 0.5f * (t0 + t1) * ray_dir;
        VolumeOut homo_out = EvalVolume(&prd->seed, m16, 0.0f, new_orig);
        //auto hg = pbrt::HenyeyGreenstein(vol_out.anisotropy);

        float3 transmittance = vec3(1.0f);
        float3 weight = vec3(1.0f); 
        float dt;

        if (sbt_data->multiscatter) {

            auto prob = prd->rndf();
            dt = -logf(1.0f-prob) / average(homo_out.extinction);

            auto pdf = expf(-homo_out.extinction * dt) * homo_out.extinction;
            weight = (homo_out.extinction * homo_out.albedo ) / pdf;

        } else {

            auto total_transmittance = expf(-homo_out.extinction * t_max);
            dt = -logf(1.0f - prd->rndf() * (1.0f - average(total_transmittance))) / average(homo_out.extinction);

            auto cdf = 1.0f - total_transmittance;
            auto pdf = expf(-homo_out.extinction * dt) * homo_out.extinction;

            weight = cdf / pdf;
            weight *= homo_out.extinction;

            //auto tmp = expf(-t_max * homo_out.density);
            //dt = -logf(1.0 - prd->rndf() *(1-tmp)) / homo_out.density;
        }

        if (prd->vol.afterSingleScatter) {

            dt = t_max;
            prd->vol.afterSingleScatter = false;
        }

        if (dt >= t_max) {
            
            cihouVolumeEdge();
            transmittance = expf(-homo_out.extinction * t_max);

        } else {

            new_orig = ray_orig + (t0+dt) * ray_dir;
            transmittance = expf(-homo_out.extinction * dt);

            if (sbt_data->multiscatter) {

                pbrt::HenyeyGreenstein hg (homo_out.anisotropy);
                float2 uu = { prd->rndf(), prd->rndf() };
                auto pdf = hg.sample(-ray_dir, new_dir, uu);              
                prd->samplePdf = pdf;
            } else {

                prd->vol.afterSingleScatter = true;
                prd->_mask_ = VolumeMatMask;
            }
        }

        prd->updateAttenuation( transmittance );

        prd->origin = new_orig;
        prd->direction = new_dir;
        prd->geometryNormal = {};

        if (dt >= t_max) { // ingore lighting
            return; 
        }

        ShadowPRD shadowPRD {};
        shadowPRD.seed = prd->seed;
        shadowPRD.origin = new_orig; //camera sapce
        shadowPRD.attanuation = vec3(1.0f);
        
        auto evalBxDF = [&](const float3& _wi_, const float3& _wo_, float& thisPDF) -> float3 {

            pbrt::HenyeyGreenstein hg(homo_out.anisotropy);
            thisPDF = hg.p(_wo_, _wi_);
            return homo_out.albedo * thisPDF;
        };

        prd->depth += 1;
        prd->lightmask = VolumeMatMask;
        DirectLighting<true>(prd, shadowPRD, new_orig+params.cam.eye, ray_dir, evalBxDF);
        //prd->radiance += prd->emission;
        prd->radiance = prd->radiance * weight;
        return;
    }

    float v_density = 0.0;
    float sigma_t = sbt_data->vol_extinction;

    VolumeOut vol_out;
    auto level = sbt_data->vol_depth;
    auto step_scale = 1.0f / sigma_t;

    float3 emitting = make_float3(0.0);
    float3 scattering = make_float3(1.0);

    float t_ele = 0;

    while(--level > 0) {
        auto prob = prd->rndf();
        auto dt = -logf(1.0f-prob) * step_scale;

        t_ele += dt;

        if (t_ele >= t_max) {

            cihouVolumeEdge();
            v_density = 0;
            break;
        } // over shoot, outside of volume

        new_orig = ray_orig + (t0+t_ele) * ray_dir;
        vol_out = EvalVolume(&prd->seed, m16, sigma_t, new_orig);

        v_density = clamp(vol_out.density / sigma_t, 0.0f, 1.0f);
        emitting += vol_out.emission;

        step_scale = fminf(step_scale, vol_out.step_scale);

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

    prd->depth += 1;
    prd->lightmask = VolumeMatMask;
    DirectLighting<true>(prd, shadowPRD, new_orig+params.cam.eye, ray_dir, evalBxDF);
    
    prd->radiance += prd->emission;
    
    return;
}

extern "C" __global__ void __anyhit__occlusion_volume()
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    ShadowPRD* prd = getPRD<ShadowPRD>();
    const float t0 = prd->vol.t0;
    const float t1 = prd->vol.t1;
    //t1 = prd->maxDistance;

    const float t_max = t1 - t0; // world space
          float t_ele = 0;

    float3 test_point = ray_orig; 
    float3 transmittance = make_float3(1.0f);
    float hgp = 1.0f;
    pbrt::HenyeyGreenstein hg(9.0f);

    float m16[16] = {}; m16[15] = 1;
    optixGetWorldToObjectTransformMatrix(m16);

    const float sigma_t = sbt_data->vol_extinction;

    if (0 == sbt_data->vol_depth) { // Homogeneous

        test_point += ray_dir * 0.5f * (t0 + t1);

        VolumeOut homo_out = EvalVolume(&prd->seed, m16, sigma_t, test_point);
        hg = pbrt::HenyeyGreenstein(homo_out.anisotropy);

        transmittance = expf(-homo_out.extinction * t_max);
        prd->attanuation *= transmittance;

        //transmittance *= vol_out.albedo * hg.p(-ray_dir, ray_dir);
        optixIgnoreIntersection();
        return;
    }

    auto level = sbt_data->vol_depth;
    while(--level > 0) {

        auto prob = prd->rndf();
        t_ele -= log(1.0f-prob) / (sigma_t);

        test_point = ray_orig + (t0+t_ele) * ray_dir;

        if (t_ele >= t_max) {
            break;
        } // over shoot, outside of volume

        VolumeOut vol_out = EvalVolume(&prd->seed, m16, sigma_t, test_point);

        const auto v_density = vol_out.density / sigma_t;

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

    if (0 == level) { transmittance = {}; }

    prd->attanuation *= transmittance;
    optixIgnoreIntersection();
    //prd->origin = ray_orig;
    //prd->direction = ray_dir;
    return;
}