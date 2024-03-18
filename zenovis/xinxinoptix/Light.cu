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

static __inline__ __device__ bool checkLightGAS(uint instanceId) {
    return ( instanceId >= params.maxInstanceID-2 );
}

static __inline__ __device__ bool isPlaneLightGAS(uint instanceId) {
    return ( instanceId == params.maxInstanceID-1 );
}

static __inline__ __device__ bool isTriangleLightGAS(uint instanceId) {
    return ( instanceId == params.maxInstanceID-2 );
}

extern "C" __global__ void __closesthit__radiance()
{
    RadiancePRD* prd = getPRD();
    
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

    bool ignore = false;
    const auto pType = optixGetPrimitiveType();

    if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {

        ignore = params.firstSphereLightIdx == UINT_MAX;
        light_index = primitiveIndex + params.firstSphereLightIdx;

    } else {
      //////

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

    bool enabled = light.mask & prd->lightmask;
    if (!enabled) { 
        prd->depth += 1;
        prd->done = true;
        return; 
    }
    
    vec3 light_normal {};

    if (pType == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_SPHERE) {
        light_normal = normalize(P - light.sphere.center);
    } else {
        light_normal = light.N;
    }

    auto visible = (light.config & zeno::LightConfigVisible);

    if (!visible && prd->depth == 0) {
        prd->_tmin_ = optixGetRayTmax();
        return;
    }

    prd->done = true;

    float3 lightDirection = optixGetWorldRayDirection(); //light_pos - P;
    float  lightDistance  = optixGetRayTmax();  //length(lightDirection);

    LightSampleRecord lsr;

    if (light.type != zeno::LightType::Diffuse) {
        if (prd->depth > 0) { return; }
        lsr.PDF = 1.0f;
        lsr.isDelta = true;
    }

    const auto lightShape = light.shape;
    const auto shadingP = ray_orig + params.cam.eye;

    const auto insideEllipse = [&]() -> bool {

        float3 _vertices_[3];
        optixGetTriangleVertexData( gas, optixGetPrimitiveIndex(), sbtGASIndex, 0, _vertices_ );
    
        const float3& v0 = _vertices_[0];
        const float3& v1 = _vertices_[1];
        const float3& v2 = _vertices_[2];

        float2 barys = optixGetTriangleBarycentrics();
        float3 P_Local = interp(barys, v0, v1, v2);

        auto delta = P_Local - light.rect.v;
            
        float2 uv = { dot(delta, light.rect.axisX) / light.rect.lenX,
                      dot(delta, light.rect.axisY) / light.rect.lenY };

        auto uvd = uv - 0.5f;
        return length(uvd) <= 0.5f;
    };

    if (prd->test_distance) {
            
        if (lightShape != zeno::LightShape::Ellipse) 
        {
            prd->maxDistance = optixGetRayTmax();
            prd->test_distance = false; return;
        }

        if (insideEllipse()) {
            prd->maxDistance = optixGetRayTmax();
            prd->test_distance = false;
        } else {
            prd->done = false;
            prd->_tmin_ = optixGetRayTmax();
        }

        return;
    }

    switch (lightShape) {
    case zeno::LightShape::Plane:
    case zeno::LightShape::Ellipse: {

        auto valid = true; 
        if (light.rect.isEllipse && !insideEllipse()) {
            valid &= false;
            prd->done = false;
            prd->_tmin_ = optixGetRayTmax();
        }
        // valid = light.rect.EvalAfterHit(&lsr, lightDirection, lightDistance, shadingP);
        // if (!valid) {
        //     prd->done = false;
        //     prd->_tmin_ = optixGetRayTmax();
        //     return;
        // };
        if (!valid) { return; }

        auto rect = light.rect;
        float2 uvScale, uvOffset;
        valid &= SpreadClampRect(rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY, 
                            rect.normal, shadingP, 
                            light.spreadMajor, uvScale, uvOffset);  

        if (!valid) { return; }

        SphericalRect squad;
        SphericalRectInit(squad, shadingP, rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY);
        lsr.PDF = 1.0f / squad.S;  
        if ( !isfinite(lsr.PDF) ) { return; }
        
        lsr.n = light.N;
        lsr.NoL = dot(light.N, -ray_dir);
        lsr.uv = uvOffset + lsr.uv * uvScale;

        lsr.dir = lightDirection;
        lsr.dist = lightDistance;
        lsr.p = ray_orig + ray_dir * lightDistance;
        break;
    }
    case zeno::LightShape::Sphere: {
        light.sphere.EvalAfterHit(&lsr, lightDirection, lightDistance, shadingP);
        cihouSphereLightUV(lsr, light); break;
    }
    case zeno::LightShape::TriangleMesh: {
    
        float2 bary2 = optixGetTriangleBarycentrics();
        float3 bary3 = { 1.0f-bary2.x-bary2.y, bary2.x, bary2.y };
        
        float3* normalBuffer = reinterpret_cast<float3*>(params.triangleLightNormalBuffer);
        float2* coordsBuffer = reinterpret_cast<float2*>(params.triangleLightCoordsBuffer);
        light.triangle.EvalAfterHit(&lsr, lightDirection, lightDistance, shadingP, prd->geometryNormal, bary3, normalBuffer, coordsBuffer);
        break;
    }
    default: return;
    }

    if (light.type == zeno::LightType::Diffuse && light.spreadMajor < 1.0f) {

        auto void_angle = 0.5f * (1.0f - light.spreadMajor) * M_PIf;
        auto atten = light_spread_attenuation(
                                lsr.dir,
                                lsr.n,
                                light.spreadMajor,
                                tanf(void_angle),
                                light.spreadNormalize);
        lsr.intensity *= atten;
    }

    if (!cihouMaxDistanceContinue(lsr, light)) { return; }
    float3 emission = cihouLightEmission(lsr, light, prd->depth);

    if (light.config & zeno::LightConfigDoubleside) {
        lsr.NoL = abs(lsr.NoL);
    }

    if (light.falloffExponent != 2.0f) {
        lsr.intensity *= powf(lsr.dist, 2.0f-light.falloffExponent);
    }

    const float _SKY_PROB_ = params.skyLightProbablity();

    if (lsr.NoL > _FLT_EPL_) {

        auto lightTree = reinterpret_cast<pbrt::LightTreeSampler*>(params.lightTreeSampler);
        if (lightTree == nullptr) { return; }

        auto PMF = lightTree->PMF(reinterpret_cast<const Vector3f&>(shadingP),
                                         reinterpret_cast<const Vector3f&>(prd->geometryNormal), light_index);

        auto lightPickPDF = (1.0f - _SKY_PROB_) * PMF;

        if (lightPickPDF < 0.0f || !isfinite(lightPickPDF)) {
            lightPickPDF = 0.0f;
        }

        if (0 == prd->depth) {
            if (light.config & zeno::LightConfigVisible) {
                prd->radiance = emission;
            }
            prd->depth = 1;
            prd->attenuation = vec3(1.0f); 
            prd->attenuation2 = vec3(1.0f);
            return;
        }
        
        float lightPDF = lightPickPDF * lsr.PDF;
        float scatterPDF = prd->samplePdf; //BxDF PDF from previous hit
        float misWeight = 1.0f;
        
        if (!lsr.isDelta) {
            misWeight = BRDFBasics::PowerHeuristic(scatterPDF, lightPDF);
        }

        prd->radiance = lsr.intensity * emission * misWeight;
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

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    // const float3 P = ray_orig + optixGetRayTmax() * ray_dir;

    auto instanceId = optixGetInstanceId();
    auto isLightGAS = checkLightGAS(instanceId);

    ShadowPRD* prd = getPRD<ShadowPRD>();

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

    if (light_index == prd->lightIdx) {
        //printf("maxDistance = %f tmax = %f \n", prd->maxDistance, optixGetRayTmax());
        ignore = true;
    }

    if (ignore) {
        optixIgnoreIntersection();
        return;
    }

    light_index = min(light_index, params.num_lights - 1);
    auto& light = params.lights[light_index];

    bool visible = (light.config & zeno::LightConfigVisible);

    if (zeno::LightShape::Ellipse == light.shape && light.rect.isEllipse) {
        LightSampleRecord lsr;
        auto rayorig = ray_orig + params.cam.eye;
        visible &= light.rect.hitAsLight(&lsr, rayorig, ray_dir);
    }

    if (visible) {
        prd->attanuation = {};
        optixTerminateRay();
    }

    optixIgnoreIntersection();
    return;
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}