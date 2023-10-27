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
#include "Sampling.h"
#include "LightTree.h"

//COMMON_CODE

static __inline__ __device__ void evalSurface(float4* uniforms) {

    //GENERATED_BEGIN_MARK

    //GENERATED_END_MARK
} 

static __inline__ __device__ bool checkLightGAS(uint instanceId) {
    return ( instanceId >= OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID-2 );
}

static __inline__ __device__ bool isPlaneLightGAS(uint instanceId) {
    return ( instanceId == OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID-1 );
}

static __inline__ __device__ bool isTriangleLightGAS(uint instanceId) {
    return ( instanceId == OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID-2 );
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

        if (isPlaneLightGAS(instanceId)) {

            ignore = params.firstRectLightIdx == UINT_MAX;
            auto rect_idx = primitiveIndex / 2;
            light_index = rect_idx + params.firstRectLightIdx;

        } else if (isTriangleLightGAS(instanceId)) {

            ignore = params.firstTriangleLightIdx == UINT_MAX;
            light_index = primitiveIndex + params.firstTriangleLightIdx;

        } else { ignore = true; }
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

    prd->depth += 1;
    prd->done = true;

    float3 lightDirection = optixGetWorldRayDirection(); //light_pos - P;
    float  lightDistance  = optixGetRayTmax();  //length(lightDirection);

    LightSampleRecord lsr;
    float3 emission = light.emission;

    if (light.type != zeno::LightType::Diffuse) {
        // auto pos = ray_orig + ray_dir * optixGetRayTmax();
        // prd->geometryNormal = normalize(light.sphere.center - ray_orig);
        // prd->offsetUpdateRay(pos, ray_dir);
        return;
    } else {
        if (light.shape == zeno::LightShape::Plane) {
            light.rect.EvalAfterHit(&lsr, lightDirection, lightDistance, prd->origin);
        } else if (light.shape == zeno::LightShape::Sphere) {
            light.sphere.EvalAfterHit(&lsr, lightDirection, lightDistance, prd->origin);

            mat3 localAxis = {
                reinterpret_cast<vec3&>(light.T), 
                reinterpret_cast<vec3&>(light.N), 
                reinterpret_cast<vec3&>(light.B) };

            auto sampleDir = localAxis * (lsr.n);
            lsr.uv = vec2(sphereUV(sampleDir, false));

        } else if (light.shape == zeno::LightShape::TriangleMesh) {

            float2 bary2 = optixGetTriangleBarycentrics();
            float3 bary3 = { 1.0f-bary2.x-bary2.y, bary2.x, bary2.y };
            
            float3* normalBuffer = reinterpret_cast<float3*>(params.triangleLightNormalBuffer);
            float2* coordsBuffer = reinterpret_cast<float2*>(params.triangleLightCoordsBuffer);
            light.triangle.EvalAfterHit(&lsr, lightDirection, lightDistance, prd->origin, prd->geometryNormal, bary3, normalBuffer, coordsBuffer);
        }
    }

    if (light.config & zeno::LightConfigDoubleside) {
        lsr.NoL = abs(lsr.NoL);
    }

    if (light.tex != 0u) {
        auto color = texture2D(light.tex, lsr.uv);
        if (light.texGamma != 1.0f) 
            color = pow(color, light.texGamma);

        if (prd->depth > 1) {
            color = color * light.intensity;
        } else {
            color = color * light.vIntensity;
        }
        emission = *(vec3*)&color;
    }

    const float _SKY_PROB_ = params.skyLightProbablity();

    if (lsr.NoL > _FLT_EPL_) {

        auto lightTree = reinterpret_cast<pbrt::LightTreeSampler*>(params.lightTreeSampler);

        auto PMF = lightTree->PMF(reinterpret_cast<const Vector3f&>(ray_orig), 
                                         reinterpret_cast<const Vector3f&>(prd->geometryNormal), light_index);

        auto lightPickPDF = (1.0f - _SKY_PROB_) * PMF;
        DCHECK(lightPickPDF > 0.0f && lightPickPDF < 1.0f);

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

        prd->radiance = emission * misWeight;
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

    bool visible = (light.config & zeno::LightConfigVisible);

    if (visible) {
        prd->shadowAttanuation = {};
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