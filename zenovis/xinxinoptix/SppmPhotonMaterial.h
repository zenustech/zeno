#pragma once

#include "SppmDevice.h"

// Photon continuation for generated material callables. The compiler can remove
// inactive lobes when graph outputs are constant; dynamic graphs use the same
// full-mixture estimator.
// Every ray-dependent value is explicit: direct callables cannot read the
// caller's optix payload registers or optixGetRayTmax().
static __forceinline__ __device__ void sppmPhotonMaterial(
    RadiancePRD& prd, MatOutput material, const MatInput& attrs, float offset,
    float3 rayDirection, float segmentLength)
{
    if (material.shadowReceiver > 0.0f && prd.depth >= 1u) {
        prd.direction = rayDirection;
        prd.alphaHit = true;
        prd._tmin_ = segmentLength;
        return;
    }

    prd.geometryNormal = attrs.wldNorm;
    if (material.doubleSide > 0.5f || material.thin > 0.5f)
        prd.geometryNormal = faceforward(prd.geometryNormal, -rayDirection, prd.geometryNormal);
    prd.done = prd.alphaDepth > 5 ? true : prd.done;
    // Preserve the random draw even for fully opaque materials. Graph opacity
    // denotes transparency here, and accepted cutouts retrace the same ray.
    if (material.opacity > rnd(prd.seed)) {
        prd.alphaHit = true;
        prd._tmin_ = segmentLength;
        prd.alphaDepth += material.emissionOnly > 0.5f ? 1 : 0;
        return;
    }
    prd._tmax_ = segmentLength;
    if (material.isHair > 0.5f || material.subsurface > 0.0f ||
        sppmMax(float3(material.emission)) > 0.0f) {
        atomicAdd(&params.sppm.statistics->unsupported, 1ull);
        prd.radiance = make_float3(0.0f);
        prd.done = true;
        return;
    }
    material.roughness = clamp(material.roughness, 0.0f, 1.0f);
    if (prd.depth == 0u && material.flatness > 0.5f) {
        prd.radiance = make_float3(0.0f);
        prd.done = true;
        return;
    }
    if (material.emissionOnly > 0.5f) {
        float3 decay = make_float3(1.0f);
        if (prd.curMatIdx > 0u) {
            vec3 sigma, alpha;
            prd.readMat(sigma, alpha);
            decay = alpha.x < 0.0f ? DisneyBSDF::Transmission(sigma, segmentLength) :
                DisneyBSDF::Transmission2(sigma * alpha, sigma, prd.channelPDF, segmentLength, true);
            prd.attenuation *= decay;
        }
        prd.radiance += decay * float3(material.emission);
        prd.done = true;
        return;
    }
    float3 tangent = attrs.T, bitangent = attrs.B;
    if (tangent == float3{} || bitangent == float3{}) {
        Onb frame(material.nrm);
        tangent = frame.m_tangent;
        bitangent = frame.m_binormal;
    }
    sppmSurface(prd, material, attrs.wldPos, prd.geometryNormal, material.nrm,
        tangent, bitangent, offset, rayDirection, segmentLength);
}
