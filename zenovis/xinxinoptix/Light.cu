#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>

#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "zeno/types/LightObject.h"
#include "zxxglslvec.h"
#include "DisneyBRDF.h"
#include "DisneyBSDF.h"

#include "Shape.h"
#include "Light.h"

//COMMON_CODE

static __inline__ __device__ void evalSurface(float4* uniforms) {

    //GENERATED_BEGIN_MARK   

    //GENERATED_END_MARK
}

static __inline__ __device__ float3 sphereUV(float3 &direction) {
    
    return float3 {
        atan2(direction.x, direction.z) / (2.0f*M_PIf) + 0.5f,
        direction.y * 0.5f + 0.5f, 0.0f
    };
} 

static __inline__ __device__ bool checkLightGAS(uint instanceId) {
    return ( instanceId >= OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID-1 );
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();
    if(prd->test_distance)
    {
        prd->vol_t1 = optixGetRayTmax();
        prd->test_distance = false; return;
    }
    
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint        primitiveIndex = optixGetPrimitiveIndex();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    // HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
    // auto zenotex = rt_data->textures;

    auto instanceId = optixGetInstanceId();
    auto isLightGAS = checkLightGAS(instanceId);

    if (params.num_lights == 0 || !isLightGAS) {
        prd->depth += 1;
        prd->done = true;
        return;
    }

    uint light_index = 0;
    vec3 light_normal {};

    bool ignore = false;
    const auto pType = optixGetPrimitiveType();

    if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {

        ignore = params.firstSphereLightIdx == UINT_MAX;
        light_index = primitiveIndex + params.firstSphereLightIdx;
    } else {

        ignore = params.firstRectLightIdx == UINT_MAX;
        auto rect_idx = primitiveIndex / 2;
        light_index = rect_idx + params.firstRectLightIdx;
    }

    if (ignore) {
        prd->depth += 1;
        prd->done = true;
        return;
    }

    light_index = min(light_index, params.num_lights - 1);
    auto& light = params.lights[light_index];

    if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {
        light_normal = normalize(light.sphere.center - ray_orig);
    } else {
        light_normal = light.N;
    }

    auto visible = (light.config & zeno::LightConfigVisible);

    if (!visible && prd->depth == 0) {
        auto pos = P;
        prd->geometryNormal = light_normal;
        prd->offsetUpdateRay(pos, ray_dir); 
        return;
    }

    float prevCDF = light_index>0? params.lights[light_index-1].CDF : 0.0f;
    float lightPickPDF = (light.CDF - prevCDF) / params.lights[params.num_lights-1].CDF;

    prd->depth += 1;
    prd->done = true;

    float3 lightDirection = optixGetWorldRayDirection(); //light_pos - P;
    float  lightDistance  = optixGetRayTmax();  //length(lightDirection);

    LightSampleRecord lsr;
    float3 emission = light.emission;

    if (light.type != zeno::LightType::Diffuse) {
        auto pos = ray_orig + ray_dir * optixGetRayTmax();
        prd->geometryNormal = normalize(light.sphere.center - ray_orig);
        prd->offsetUpdateRay(pos, ray_dir);
        return;
    } else {
        if (light.shape == zeno::LightShape::Plane) {
            light.rect.eval(&lsr, lightDirection, lightDistance, prd->origin);
        } else if (light.shape == zeno::LightShape::Sphere){
            light.sphere.eval(&lsr, lightDirection, lightDistance, prd->origin);
        }
    }

    if (light.config & zeno::LightConfigDoubleside) {
        lsr.NoL = abs(lsr.NoL);
    }

    if (lsr.NoL > _FLT_EPL_) {

        if (1 == prd->depth) {
            if (light.config & zeno::LightConfigVisible) {
                prd->radiance = emission;
            }
            prd->attenuation = vec3(1.0f); 
            prd->attenuation2 = vec3(1.0f);
            return;
        }
        
        float lightPDF = lightPickPDF * lsr.PDF;
        float scatterPDF = prd->samplePdf; //BxDF PDF from previous hit
        float misWeight = BRDFBasics::PowerHeuristic(scatterPDF, lightPDF);

        prd->radiance = light.emission * misWeight;
        // if (scatterPDF > __FLT_DENORM_MIN__) {
        //     prd->radiance /= scatterPDF;
        // }
    }
    return;
}

extern "C" __global__ void __anyhit__shadow_cutout()
{
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const uint           sbtGASIndex = optixGetSbtGASIndex();
    const uint          primitiveIdx = optixGetPrimitiveIndex();

    // const float3 ray_orig = optixGetWorldRayOrigin();
    // const float3 ray_dir  = optixGetWorldRayDirection();
    // const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    auto instanceId = optixGetInstanceId();
    auto isLightGAS = checkLightGAS(instanceId);

    RadiancePRD* prd = getPRD();

    if (params.num_lights == 0 || !isLightGAS) {
        optixIgnoreIntersection();
        return;
    }

    uint light_index = 0;
    vec3 light_normal {};

    bool ignore = false;
    const auto pType = optixGetPrimitiveType();

    if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {

        ignore = params.firstSphereLightIdx == UINT_MAX;
        light_index = primitiveIdx + params.firstSphereLightIdx;
    } else {

        ignore = params.firstRectLightIdx == UINT_MAX;
        auto rect_idx = primitiveIdx / 2;
        light_index = rect_idx + params.firstRectLightIdx;
    }

    if (ignore) {
        optixIgnoreIntersection();
        return;
    }

    light_index = min(light_index, params.num_lights - 1);
    auto& light = params.lights[light_index];

    auto visible = (light.config & zeno::LightConfigVisible);

    if (visible) {
        prd->attenuation2 = {};
        prd->attenuation = {};
        optixTerminateRay();
    }

    optixIgnoreIntersection();
    return;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}