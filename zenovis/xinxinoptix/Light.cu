#include <optix.h>
#include <cuda/random.h>
#include <cuda/helpers.h>

#include "optixPathTracer.h"
#include "TraceStuff.h"
#include "zeno/types/LightObject.h"
#include "zxxglslvec.h"
#include "DisneyBRDF.h"

#include "Shape.h"
#include "Light.h"
#include "Sampling.h"
#include "LightTree.h"

static __inline__ __device__ bool isPlaneLightGAS(uint instanceId) {
    return ( instanceId == 1 );
}

static __inline__ __device__ bool isTriangleLightGAS(uint instanceId) {
    return ( instanceId == 0 );
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

    if (params.num_lights == 0) {
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

	const auto rectUV = [&]() -> float2 {
		float2 uv0, uv1, uv2;
		if (primitiveIndex % 2 == 0) {
			uv0 = float2{0, 0};
			uv1 = float2{1, 0};
			uv2 = float2{1, 1};
		} else {
			uv0 = float2{0, 0};
			uv1 = float2{1, 1};
			uv2 = float2{0, 1};
		}

		float2 barys = optixGetTriangleBarycentrics();
		return interp(barys, uv0, uv1, uv2);
	};

	const auto insideEllipse = [&]() -> bool {
		float2 uv = rectUV();
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
	case zeno::LightShape::Ellipse: {
		
		if (light.rect.isEllipse && !insideEllipse()) {
			prd->done = false;
			prd->_tmin_ = optixGetRayTmax();
			return;
		}
	}
	case zeno::LightShape::Plane: {
    
		lsr.uv = rectUV();

		if (prd->depth > 0) {

			auto rect = light.rect;
			float2 uvScale, uvOffset;
			bool valid = SpreadClampRect(rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY, rect.normal, 
											shadingP, light.spreadMajor, uvScale, uvOffset);
			// if (!valid) { return; }

			SphericalRect squad;
			SphericalRectInit(squad, shadingP, rect.v, rect.axisX, rect.lenX, rect.axisY, rect.lenY);
			lsr.PDF = 1.0f / squad.S;
			if (!isfinite(lsr.PDF)) {
				return;
			}
			lsr.uv = uvOffset + lsr.uv * uvScale;
		}

		lsr.n = light.N;
		lsr.NoL = dot(light.N, -ray_dir);

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

    if (light.type == zeno::LightType::Diffuse && light.spreadMajor < 1.0f && prd->depth > 0) {

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
        auto tmp = float3{1, 1, 1};
        prd->updateAttenuation(tmp);
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

    ShadowPRD* prd = getPRD<ShadowPRD>();

    if (params.num_lights == 0) {
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
