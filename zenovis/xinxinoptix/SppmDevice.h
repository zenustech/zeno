#pragma once

#include "DisneyBSDF.h"
#include "Light.h"
#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>

static __forceinline__ __device__ bool sppmFinite(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
static __forceinline__ __device__ float sppmMax(float3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

#define SPPM_MATERIAL_FIELDS(F) \
    F(basecolor) F(transColor) F(diffractColor) F(roughness) F(anisotropic) \
    F(anisoRotation) F(ior) F(metallic) F(specular) F(specularTint) F(sheen) \
    F(sheenTint) F(clearcoat) F(clearcoatRoughness) F(clearcoatIOR) \
    F(specTrans) F(F0) F(diffraction) F(thin)

static __forceinline__ __device__ SppmMaterial sppmPack(const MatOutput& m) {
    SppmMaterial result;
#define SPPM_COPY(field) result.field = m.field;
    SPPM_MATERIAL_FIELDS(SPPM_COPY)
#undef SPPM_COPY
    return result;
}
static __forceinline__ __device__ MatOutput sppmUnpack(const SppmMaterial& m) {
    MatOutput result = {};
#define SPPM_COPY(field) result.field = m.field;
    SPPM_MATERIAL_FIELDS(SPPM_COPY)
#undef SPPM_COPY
    return result;
}
#undef SPPM_MATERIAL_FIELDS

// Evaluate camera-side transport with the same density as the VNDF sampler.
static __forceinline__ __device__ float3 sppmEvaluate(
    const MatOutput& material, float3 wi, float3 wo, float3 tangent,
    float3 bitangent, float3 shadingNormal, float3 normal, float& pdf, RadiancePRD* prd = nullptr)
{
    if (material.specTrans==0.0f && material.isHair<=0.5f && material.subsurface==0.0f)
        return DisneyBSDF::EvaluateOpaqueSppm(material,wi,wo,tangent,bitangent,shadingNormal,pdf);
    float reversePdf = 0.0f;
    vec3 d = {}, s = {}, t = {};
    float3 f = DisneyBSDF::EvaluateDisney3<true>(vec3(1.0f), material, wi, wo,
        tangent, bitangent, shadingNormal, normal, material.thin > 0.5f,
        false, pdf, reversePdf, dot(shadingNormal, wi), d, s, t, false, false, prd);
    if (material.thin < 0.5f && dot(wi, shadingNormal) * dot(wo, shadingNormal) < 0.0f) {
        const float eta = dot(wo, shadingNormal) > 0.0f ? material.ior : 1.0f / material.ior;
        f /= eta * eta;
    }
    return f;
}

// Accumulate each photon once per visible receiver within its gather sphere.
static __forceinline__ __device__ void sppmAccumulate(uint32_t index,
    float3 position, float3 normal, float3 wi, float3 beta, RadiancePRD& prd)
{
    const auto& launch = params.sppm;
    auto& pixel = launch.pixels[index];
    if (!pixel.visible || !(lengthSquared(pixel.position-position) <= pixel.radius*pixel.radius) ||
        !(dot(pixel.normal, normal) > 0.0f)) return;
    // EvaluateDisney3 returns f * |n.wi|. Photon density estimates
    // require f alone; applying another cosine would darken them.
    const float cosine = fabsf(dot(pixel.shadingNormal, wi));
    if (cosine > 1.0e-6f) {
        float pdf = 0.0f;
        const auto material = sppmUnpack(pixel.material);
        float3 f = sppmEvaluate(material,
            wi, pixel.wo, pixel.tangent, pixel.bitangent, pixel.shadingNormal,
            pixel.normal, pdf, &prd);
        const float3 phi = pixel.beta * beta * f / cosine;
        if (sppmFinite(phi)) {
            if (launch.guideCells) {
                // Image-space second moment of the unweighted
                // emission integrand. The emission PDF correction
                // is applied once per path when training the guide.
                const double r2 = double(pixel.radius)*pixel.radius;
                prd.sppmGuideScore += (double(phi.x)*phi.x +
                    double(phi.y)*phi.y + double(phi.z)*phi.z)/(r2*r2);
            }
            atomicAdd(&pixel.phi.x, phi.x);
            atomicAdd(&pixel.phi.y, phi.y);
            atomicAdd(&pixel.phi.z, phi.z);
            atomicAdd(&pixel.m, 1u);
            ++prd.sppmGathered;
            if (prd.sppmGatherSpecular) {
                ++prd.sppmCaustics;
                if (sppmMax(phi) > 0.0f) prd.sppmGuideUseful = 1u;
            }
        }
    }
}

static __forceinline__ __device__ void sppmGather(
    float3 position, float3 normal, float3 wi, float3 beta, RadiancePRD& prd)
{
    const auto& launch = params.sppm;
    const int3 cell = sppmCell(position, launch.initialRadius);
    for (int z=-1; z<=1; ++z)
    for (int y=-1; y<=1; ++y)
    for (int x=-1; x<=1; ++x) {
        const int3 query = make_int3(cell.x+x, cell.y+y, cell.z+z);
        uint32_t index = launch.heads[sppmHash(query, launch.hashSize)];
        while (index != UINT_MAX) {
            const auto& pixel = launch.pixels[index];
            if (pixel.cell.x==query.x && pixel.cell.y==query.y && pixel.cell.z==query.z)
                sppmAccumulate(index, position, normal, wi, beta, prd);
            index = pixel.next;
        }
    }
}

// SPPM has its own surface transport so the interactive path tracer's
// throughput/firefly clamps and diffuse roughness changes cannot bias flux.
template<bool Photon, bool DeferredCamera=false>
static __forceinline__ __device__ void sppmSurfaceMode(
    RadiancePRD& prd, MatOutput material, float3 position, float3 normal,
    float3 shadingNormal, float3 tangent, float3 bitangent, float offset,
    float3 rayDirection, float segmentLength = optixGetRayTmax())
{
    constexpr bool photon=Photon;
    const float3 wo = -normalize(rayDirection);
    if (!photon && prd.depth == 0u) {
        if (params.sppm.iteration == 0u && params.sppm.radiusPixels > 0.0f &&
            prd.sppmCameraSample == 0u) {
            const float depth = params.physical_camera_panorama_camera ? length(position) :
                dot(position, normalize(params.cam.front));
            if (isfinite(depth) && depth > 0.0f) {
                atomicMin(&params.sppm.statistics->cameraDepthBits, __float_as_uint(depth));
                const bool caster = material.specTrans > 0.0f || material.metallic > 0.0f ||
                    (material.roughness < 0.2f && (material.specular > 0.0f || material.clearcoat > 0.0f));
                if (caster) atomicMin(&params.sppm.statistics->casterDepthBits, __float_as_uint(depth));
            }
        }
        // The independent pilot requests at least four display samples on
        // primary glass. Only replica zero supplies the photon visible point.
        prd.sppmCameraSamples = material.specTrans > 0.0f ? 4u : 1u;
        if (prd.sppmCameraSample != 0u)
            prd.seed = pcg_hash(prd.seed ^ tea<4>(prd.sppmCameraSample, params.sppm.iteration));
    }
    if (material.isHair > 0.5f || material.subsurface > 0.0f) {
        atomicAdd(&params.sppm.statistics->unsupported, 1ull);
        prd.done = true;
        prd.radiance = make_float3(0.0f);
        return;
    }

    float3 segmentWeight = make_float3(1.0f);
    if (prd.curMatIdx > 0u) {
        vec3 sigma, alpha;
        prd.readMat(sigma, alpha);
        segmentWeight = DisneyBSDF::Transmission(sigma, segmentLength);
        prd.attenuation *= segmentWeight;
    }
    if (photon && prd.depth == 0u && prd.sppmLight < params.num_lights) {
        auto& light = params.lights[prd.sppmLight];
        const float distance = segmentLength;
        const float d2 = distance * distance;
        // Match the source's existing near-field model on the first segment.
        // Later segments transport the resulting flux without this factor.
        if (light.type == zeno::LightType::IES && light.ies != 0u) {
            const float radius = light.shape == zeno::LightShape::Sphere ? light.sphere.radius : 0.0f;
            const float centerD2 = (distance + radius) * (distance + radius);
            prd.attenuation *= centerD2 * PointIntensity(centerD2, distance, 1.0f);
        } else if (light.type == zeno::LightType::Spot) {
            prd.attenuation *= d2 * PointIntensity(d2, distance, 1.0f);
        } else if (light.type == zeno::LightType::Projector) {
            const float axial = distance * dot(normalize(rayDirection), light.N);
            prd.attenuation *= axial * axial * PointIntensity(axial * axial, axial, 1.0f);
        } else if (light.shape == zeno::LightShape::Point) {
            prd.attenuation *= d2 * PointIntensity(d2, distance, light.point.r * light.point.r);
        } else if (light.type == zeno::LightType::Direction && light.shape == zeno::LightShape::Sphere) {
            const float centerD = distance + light.sphere.radius;
            prd.attenuation *= centerD * centerD / fmaxf(d2, 1.0e-20f);
        }
        if (distance >= light.maxDistance) { prd.done = true; return; }
        if (light.maxDistance < FLT_MAX) {
            const float delta = 1.0f - distance / light.maxDistance;
            prd.attenuation *= smoothstep(0.0f, 1.0f, delta);
        }
        if (light.falloffExponent != 2.0f)
            prd.attenuation *= powf(fmaxf(distance, 1.0e-6f), 2.0f - light.falloffExponent);
    }

    if (photon && prd.depth > 0u) {
        // Defer collection until the surface trace has returned. Both search
        // backends then leave the transport shader and its numerical path identical.
        prd.sppmGatherPosition = position;
        prd.sppmGatherNormal = normal;
        prd.sppmGatherWi = wo;
        prd.sppmGatherBeta = prd.attenuation;
        prd.sppmGatherSpecular = prd.sppmSpecular;
        prd.sppmGatherPending = 1u;
    }


    const bool diffuse = (1.0f - material.metallic) * (1.0f - material.specTrans) > 0.0f;
    const bool glossy = (material.roughness > 0.01f && (material.specular > 0.0f ||
        material.metallic > 0.0f || material.specTrans > 0.0f)) ||
        (material.clearcoat > 0.0f && material.clearcoatRoughness > 0.01f);
    const bool gather = !photon && (diffuse || (glossy && prd.depth + 1u >= prd.max_depth));
    float3 front, back;
    SelfIntersectionAvoidance::offsetSpawnPoint(front, back, position, normal, offset);
    if constexpr (!Photon) {
    auto eval = [&](const float3& wi, const float3& outgoing, float& pdf) -> float3 {
        return sppmEvaluate(material, wi, outgoing, tangent, bitangent,
            shadingNormal, normal, pdf, &prd);
    };
    prd.radiance = float3(material.emission);
    if (!photon) {
        ShadowPRD shadow = {};
        shadow.seed = prd.seed;
        shadow.depth = prd.depth;
        shadow.origin = dot(wo, normal) >= 0.0f ? front : back;
        shadow.ShadowNormal = dot(wo, normal) >= 0.0f ? normal : -normal;
        shadow.oppositeOrigin = dot(wo, normal) >= 0.0f ? back : front;
        shadow.surfaceTransmission = material.specTrans > 0.0f;
        shadow.attanuation = make_float3(1.0f);
        if constexpr (DeferredCamera) {
            if (gather) DirectLighting<false,false,true>(shadow,shadow.origin+params.cam.eye,rayDirection,
                eval,(void*)nullptr,(void*)nullptr,prd.sppmCameraShadow);
            else DirectLighting<true,false,true>(shadow,shadow.origin+params.cam.eye,rayDirection,
                eval,(void*)nullptr,(void*)nullptr,prd.sppmCameraShadow);
            auto& request=*prd.sppmCameraShadow;
            request.state=shadow;
            // Visibility has its own random domain; BSDF continuation does not
            // depend on how many cutout intersections the shadow ray visits.
            request.state.seed=pcg_hash(shadow.seed^0xb5297a4du);
            request.radiance=shadow.radiance*segmentWeight;
        } else {
            if (gather) DirectLighting<false>(shadow,shadow.origin+params.cam.eye,rayDirection,eval);
            else DirectLighting<true>(shadow,shadow.origin+params.cam.eye,rayDirection,eval);
            prd.radiance+=shadow.radiance;
        }
        prd.radiance *= segmentWeight;
        prd.seed = shadow.seed;
    }
    } else prd.radiance=make_float3(0);

    if (gather) {
        if (prd.sppmCameraSample != 0u) { prd.done = true; return; }
        auto& pixel = params.sppm.pixels[prd.sppmPixel];
        pixel.position = position;
        pixel.normal = normal;
        pixel.shadingNormal = shadingNormal;
        pixel.tangent = tangent;
        pixel.bitangent = bitangent;
        pixel.wo = wo;
        pixel.beta = prd.attenuation;
        pixel.material = sppmPack(material);
        pixel.visible = 1u;
        atomicAdd(&params.sppm.statistics->visible, 1ull);

        prd.done = true;
        return;
    }

    bool inside = false, isDiffuse = false, isSS = false, isTransmission = false;
    vec3 wi = {}, value = {}, extinction = {};
    float forwardPdf = 0.0f, reversePdf = 0.0f;
    DisneyBSDF::SurfaceEventFlags flag = DisneyBSDF::scatterEvent;
    const bool sampled = DisneyBSDF::SampleDisney3<true>(prd.seed, prd.eventseed, material,
        tangent, bitangent, shadingNormal, normal, wo, material.thin > 0.5f,
        inside, wi, value, reversePdf, forwardPdf, flag, prd.medium, extinction,
        isDiffuse, isSS, isTransmission, prd.minSpecRough, &prd);
    if (!sampled || !(forwardPdf > 0.0f) || !sppmFinite(wi) || lengthSquared(wi) == 0.0f) {
        prd.done = true;
        return;
    }
    float3 weight = float3(value) / forwardPdf;
    const bool transmission = dot(float3(wi), normal) * dot(wo, normal) < 0.0f;
    if (transmission && material.thin < 0.5f) {
        const bool entering = dot(wo, normal) > 0.0f;
        const float eta = entering ? material.ior : 1.0f / material.ior;
        // The SPPM sampler returns importance transport. Camera radiance needs
        // eta^-2; photon paths retain unit flux through a lossless interface.
        if (!photon) {
            weight /= eta * eta;
            prd.sppmEtaScale *= eta * eta;
        }
        if (entering) prd.pushMat(extinction, vec3(-1.0f));
        else if (prd.curMatIdx > 0u) { vec3 sigma, alpha; prd.popMat(sigma, alpha); }
    }
    if (photon) {
        const float denominator = dot(wo, normal) * dot(float3(wi), shadingNormal);
        if (fabsf(denominator) < 1.0e-8f) { prd.done = true; return; }
        weight *= fabsf(dot(wo, shadingNormal) * dot(float3(wi), normal) / denominator);
    }
    if (!sppmFinite(weight)) { prd.done = true; return; }
    prd.sppmSpecular = !isDiffuse;
    prd.attenuation *= weight;
    prd.direction = normalize(float3(wi));
    prd.origin = dot(prd.direction, normal) >= 0.0f ? front : back;
    // The minimum roughness still samples a finite GGX VNDF, not a delta.
    // Its actual solid-angle PDF must compete with next-event estimation.
    prd.samplePdf = transmission && material.thin > 0.5f ? 0.0f : forwardPdf;
    prd.medium = DisneyBSDF::vacuum;
    prd.maxDistance = FLT_MAX;
    prd._mask_ = EverythingMask ^ VolumeMatMask;
    prd._tmin_ = 0.0f;
    ++prd.depth;
}

static __forceinline__ __device__ void sppmSurface(
    RadiancePRD& prd,MatOutput material,float3 position,float3 normal,float3 shadingNormal,
    float3 tangent,float3 bitangent,float offset,float3 rayDirection,float segmentLength=optixGetRayTmax()) {
#if defined(SPPM_PHOTON_PROGRAM)
    sppmSurfaceMode<true>(prd,material,position,normal,shadingNormal,tangent,bitangent,offset,rayDirection,segmentLength);
#elif defined(SPPM_CAMERA_PROGRAM)
    sppmSurfaceMode<false>(prd,material,position,normal,shadingNormal,tangent,bitangent,offset,rayDirection,segmentLength);
#else
    if (prd.sppmMode==2u)
        sppmSurfaceMode<true>(prd,material,position,normal,shadingNormal,tangent,bitangent,offset,rayDirection,segmentLength);
    else sppmSurfaceMode<false>(prd,material,position,normal,shadingNormal,tangent,bitangent,offset,rayDirection,segmentLength);
#endif
}
