#pragma once

#include <optix.h>
#include <optix_device.h>
#include <sutil/vec_math.h>

#include "DisneyBRDF.h"
#include "DisneyBSDF.h"
#include "IOMat.h"
#include "Light.h"
#include "TraceStuff.h"
#include "MneeMath.h"
#include "MneeOptix.h"

namespace Mnee {

static __forceinline__ __device__
bool traceProbe(const float3& origin, const float3& direction, float tmin, float tmax, Hit& hit)
{
    unsigned int payload0 = 0u;
    unsigned int payload1 = 0u;
    optixTraverse(params.handle, origin, direction, tmin, tmax, 0.0f,
                  DefaultMatMask, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                  RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
                  payload0, payload1);
    return captureOutgoingHit(hit);
}

static __forceinline__ __device__ float advanceT(float t)
{
    return t + fmaxf(0.0001f, fabsf(t) * 0.00001f);
}

template <bool EvaluateMaterial = true, typename VertexType>
static __forceinline__ __device__ bool
mnee_setup_manifold_vertex(Hit& hit, const float3& rayDirection, uint32_t seed, VertexType& vertex, const ManifoldMaterial* cachedMaterial = nullptr)
{
    if (optixGetPrimitiveType(hit.hitKind) != OPTIX_PRIMITIVE_TYPE_TRIANGLE ||
        hit.sbtData == nullptr || !hit.sbtData->causticCaster || hit.sbtData->opacity != 1.0f)
    {
        return false;
    }
    if (!outgoingHitTransforms(hit.objectToWorld, hit.worldToObject, hit.gas)) {
        return false;
    }

    float3 objectVertices[3];
    optixGetTriangleVertexData(hit.gas, hit.primitiveIndex, hit.sbtGasIndex, 0.0f, objectVertices);

    const auto gasPointers = reinterpret_cast<void**>(optixGetGASPointerFromHandle(hit.gas));
    if (gasPointers == nullptr) {
        return false;
    }
    const auto indexBuffer = reinterpret_cast<const uint3*>(*(gasPointers - 1));
    const auto normalBuffer = reinterpret_cast<const ushort3*>(*(gasPointers - 4));
    if (indexBuffer == nullptr || normalBuffer == nullptr) {
        return false;
    }

    const uint3 vertexIndices = indexBuffer[hit.primitiveIndex];
    const float3 objectNormals[3] = {
        decodeHalf(normalBuffer[vertexIndices.x]),
        decodeHalf(normalBuffer[vertexIndices.y]),
        decodeHalf(normalBuffer[vertexIndices.z])};

    float3 worldVertices[3];
    float3 worldNormals[3];
    for (int i = 0; i < 3; ++i) {
        worldVertices[i] = transformPoint(objectVertices[i], hit.objectToWorld);
        worldNormals[i] = normalize(transformNormal(objectNormals[i], hit.worldToObject));
    }

    // Preserve outward normals for negative-determinant instance transforms.
    const float3 axisX = make_float3(hit.objectToWorld[0].x, hit.objectToWorld[1].x, hit.objectToWorld[2].x);
    const float3 axisY = make_float3(hit.objectToWorld[0].y, hit.objectToWorld[1].y, hit.objectToWorld[2].y);
    const float3 axisZ = make_float3(hit.objectToWorld[0].z, hit.objectToWorld[1].z, hit.objectToWorld[2].z);
    if (!mnee_setup_manifold_geometry(worldVertices, worldNormals, hit.barys, vertex,
                                      dot(cross(axisX, axisY), axisZ) < 0.0f)) {
        return false;
    }

    vertex.instanceId = hit.casterId;
    if constexpr (!EvaluateMaterial) {
        // Walk the manifold for the seed closure. Tentative Newton points need
        // geometry, not another complete material-graph evaluation. Re-evaluate
        // the closure at the final visible path before computing its weight.
        return cachedMaterial != nullptr && cachedMaterial->sourceSbt == hit.sbtData;
    }
    else {
        const float bary0 = 1.0f - hit.barys.x - hit.barys.y;
        MatInput attrs = {};
        attrs.ptype = OPTIX_PRIMITIVE_TYPE_TRIANGLE;
        attrs.gas = hit.gas;
        attrs.sbtIdx = hit.sbtGasIndex;
        attrs.priIdx = hit.primitiveIndex;
        attrs.instId = hit.instanceId;
        attrs.instIdx = hit.instanceIndex;
        attrs.rayLength = hit.t;
        attrs.isBackFace = optixIsBackFaceHit(hit.hitKind);
        attrs.isShadowRay = false;
        attrs.seed = seed;
        attrs.barys2 = hit.barys;
        attrs.vertices = objectVertices;
        attrs.vertex_idx = vertexIndices;
        attrs.objPos = bary0 * objectVertices[0] + hit.barys.x * objectVertices[1] + hit.barys.y * objectVertices[2];
        attrs.objNorm = normalize(cross(objectVertices[1] - objectVertices[0],
                                        objectVertices[2] - objectVertices[0]));
        attrs.wldPos = vertex.p;
        attrs.wldNorm = vertex.ng;
        attrs.ray = rayDirection;
        attrs.V = -rayDirection;
        for (int i = 0; i < 3; ++i) {
            attrs.objectToWorld[i] = hit.objectToWorld[i];
            attrs.worldToObject[i] = hit.worldToObject[i];
        }
        attrs.N = reinterpret_cast<const TriangleInput&>(attrs).interpNorm();
        attrs.T = reinterpret_cast<const TriangleInput&>(attrs).interpTang();
        if (length(attrs.T) <= 0.0f) {
            float3 tangent;
            float3 bitangent;
            makeOrthonormals(attrs.N, tangent, bitangent);
            attrs.T = tangent;
            attrs.B = bitangent;
        }
        else {
            attrs.B = cross(attrs.T, attrs.N);
        }

        const MatOutput material = optixDirectCall<MatOutput, cudaTextureObject_t[], MatInput&>(hit.sbtData->dc_index, hit.sbtData->textures, attrs);

        const float glassWeight = (1.0f - saturate(material.metallic)) * saturate(material.specTrans);
        if (material.thin > 0.5f || material.smoothness < 0.999f || material.roughness > SmoothRoughnessLimit || material.ior <= 1.0001f || glassWeight <= 0.0001f || material.diffraction > 0.0001f)
        {
            return false;
        }

        vertex.instanceId = hit.casterId;
        vertex.material.eta = material.ior;
        vertex.material.specular = material.specular;
        vertex.material.f0 = material.F0;
        vertex.material.glassWeight = glassWeight;
        vertex.material.transmissionColor =
            make_float3(sqrtf(fmaxf(0.0f, material.transColor.x)),
                        sqrtf(fmaxf(0.0f, material.transColor.y)),
                        sqrtf(fmaxf(0.0f, material.transColor.z)));
        vertex.material.extinction = DisneyBSDF::CalculateExtinction(material.transTint, material.transTintDepth);
        vertex.material.sourceSbt = hit.sbtData;
        return true;
    }
}

static __forceinline__ __device__ bool projectToInstance(
    const float3& origin,
    const float3& target,
    uint32_t instanceId,
    const ManifoldMaterial& material,
    uint32_t seed,
    ManifoldGeometry& projected)
{
    float3 direction = target - origin;
    float distance;
    if (!normalizeWithLength(direction, distance, false)) {
        return false;
    }

    const float tmax = distance * MNEE_PROJECTION_DISTANCE_MULTIPLIER;
    float tmin = fmaxf(MNEE_MIN_PROGRESS_DISTANCE, distance * 0.000001f);
    for (int intersectionIndex = 0;
         intersectionIndex < MNEE_MAX_INTERSECTION_COUNT && tmin < tmax;
         ++intersectionIndex)
    {
        Hit hit;
        if (!traceProbe(origin, direction, tmin, tmax, hit)) {
            return false;
        }
        if (hit.casterId == instanceId) {
            return mnee_setup_manifold_vertex<false>(hit, direction, seed, projected, &material);
        }
        tmin = advanceT(hit.t);
    }
    return false;
}

static __forceinline__ __device__
float3 mnee_eval_bsdf_contribution(const ManifoldVertex& vertex, const float3& wi, const float3& wo)
{
    const float eta = effectiveEta(vertex, wi);
    float3 halfVector = -(wi + eta * wo);
    const float halfLength = length(halfVector);
    if (!(halfLength > MNEE_MIN_DISTANCE)) {
        return {};
    }
    halfVector /= halfLength;

    const float cosNormalIncoming = dot(vertex.n, wi);
    const float cosHalfIncoming = dot(halfVector, wi);
    const float cosThetaMicro = dot(vertex.n, halfVector);
    const float denominator = cosNormalIncoming * cosThetaMicro * cosThetaMicro;
    if (fabsf(denominator) < MNEE_MIN_DETERMINANT) {
        return {};
    }

    const float fresnel = BRDFBasics::DielectricFresnel(fabsf(cosHalfIncoming), eta);
    const float fresnelWeight = mix(vertex.material.f0, 1.0f, fresnel) *
                                vertex.material.specular;
    const float transmission = saturate(1.0f - fresnelWeight);
    const float scalar = vertex.material.glassWeight * transmission *
                         fabsf(cosHalfIncoming / denominator);
    return vertex.material.transmissionColor * scalar;
}

static __forceinline__ __device__ bool validateSegment(
    const float3& origin, const float3& endpoint, uint32_t expectedInstanceId, Hit& hit)
{
    float3 direction = endpoint - origin;
    float expectedDistance;
    if (!normalizeWithLength(direction, expectedDistance, false)) {
        return false;
    }

    const float tolerance = fmaxf(MNEE_MIN_DISTANCE, expectedDistance * 0.001f);
    if (!traceProbe(origin,
                    direction,
                    fmaxf(MNEE_MIN_PROGRESS_DISTANCE, expectedDistance * 0.000001f),
                    expectedDistance + tolerance,
                    hit))
    {
        return false;
    }
    return hit.casterId == expectedInstanceId && fabsf(hit.t - expectedDistance) <= tolerance;
}

static __forceinline__ __device__ float finiteLightPdfArea(const LightSampleRecord& sample)
{
    if (sample.isDelta) {
        return sample.PDF;
    }
    const float distanceSquared = sample.dist * sample.dist;
    return distanceSquared > 0.0f
               ? sample.PDF * fabsf(sample.NoL) / distanceSquared
               : 0.0f;
}

static __forceinline__ __device__ bool mnee_path_contribution(
    const float3& receiverPosition,
    const LightSampleRecord& lightSample,
    const float3& lightEmission,
    bool lightFixedDirection,
    bool lightDoubleSided,
    uint32_t seed,
    const ShadowPRD& baseShadow,
    float3& receiverDirection,
    float3& illumination,
    int vertexCount,
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS])
{
    if (vertexCount < 1 || vertexCount > MNEE_MAX_CAUSTIC_CASTERS) {
        return false;
    }
    const bool lightInfinite = lightSample.dist == FLT_MAX;
    // Convert the light sample measure through the constrained interface chain.
    if (!mnee_compute_constraint_derivatives(vertexCount,
                                      vertices,
                                      receiverPosition,
                                      lightFixedDirection,
                                      lightFixedDirection ? lightSample.dir : lightSample.p))
    {
        return false;
    }

    float firstVertexToLight = 0.0f;
    if (!mnee_compute_transfer_matrix(receiverPosition,
                               lightSample,
                               lightFixedDirection,
                               vertexCount,
                               vertices,
                               firstVertexToLight))
    {
        return false;
    }

    float receiverDistance;
    receiverDirection = vertices[0].p - receiverPosition;
    if (!normalizeWithLength(receiverDirection, receiverDistance, false)) {
        return false;
    }

    const float receiverGeometry = fminf(
        fabsf(dot(receiverDirection, vertices[0].n)) /
            (receiverDistance * receiverDistance) * firstVertexToLight,
        2.0f);
    if (!(receiverGeometry > 0.0f) || !isfinite(receiverGeometry)) {
        return false;
    }

    float lightPdf = lightFixedDirection ? lightSample.PDF : finiteLightPdfArea(lightSample);
    if (!(lightPdf > 1.0e-8f) || !isfinite(lightPdf)) {
        return false;
    }
    illumination = lightEmission * (receiverGeometry / lightPdf);

    // Verify each receiver-to-caster segment, re-evaluate its final material,
    // and track the active glass medium for Beer attenuation.
    float3 previousPosition = receiverPosition;
    float3 mediumExtinction[MNEE_MAX_CAUSTIC_CASTERS];
    uint32_t mediumInstances[MNEE_MAX_CAUSTIC_CASTERS];
    int mediumCount = 0;
    for (int i = 0; i < vertexCount; ++i) {
        ManifoldVertex& vertex = vertices[i];
        Hit finalHit;
        if (!validateSegment(previousPosition, vertex.p, vertex.instanceId, finalHit)) {
            illumination = {};
            return true;
        }

        const float seedEta = vertex.material.eta;
        if (!mnee_setup_manifold_vertex(finalHit, normalize(vertex.p - previousPosition), seed, vertex) ||
            fabsf(vertex.material.eta - seedEta) > 0.001f)
        {
            illumination = {};
            return true;
        }
        const float3 wi = normalize(previousPosition - vertex.p);
        const float3 wo = normalize(i == vertexCount - 1
                                        ? (lightFixedDirection ? lightSample.dir
                                                               : lightSample.p - vertex.p)
                                        : vertices[i + 1].p - vertex.p);
        const float3 vertexContribution = mnee_eval_bsdf_contribution(vertex, wi, wo);
        if (length(vertexContribution) <= 0.0f) {
            illumination = {};
            return true;
        }
        const bool entering = dot(wi, vertex.ng) > 0.0f;
        // Match PT's Beer attenuation over the actual refracted segment.
        // If the first interface is an exit, the receiver starts inside glass.
        const float3 extinction = mediumCount > 0 ? mediumExtinction[mediumCount - 1]
            : (!entering ? vertex.material.extinction : make_float3(0.0f));
        illumination *= float3(DisneyBSDF::Transmission(extinction, length(vertex.p - previousPosition)));
        illumination *= vertexContribution;

        if (entering) {
            mediumExtinction[mediumCount] = vertex.material.extinction;
            mediumInstances[mediumCount++] = vertex.instanceId;
        }
        else {
            for (int medium = mediumCount - 1; medium >= 0; --medium) {
                if (mediumInstances[medium] == vertex.instanceId) {
                    for (int next = medium + 1; next < mediumCount; ++next) {
                        mediumExtinction[next - 1] = mediumExtinction[next];
                        mediumInstances[next - 1] = mediumInstances[next];
                    }
                    --mediumCount;
                    break;
                }
            }
        }
        previousPosition = vertex.p;
    }

    // Resolve the outgoing endpoint. A finite directional emitter constrains
    // both direction and footprint; an environment endpoint has no finite range.
    const ManifoldVertex& lastVertex = vertices[vertexCount - 1];
    float3 lightDirection = lightFixedDirection ? lightSample.dir : lightSample.p - lastVertex.p;
    float lightDistance;
    if (lightInfinite) {
        lightDistance = FLT_MAX;
        lightDirection = normalize(lightDirection);
    }
    else if (lightFixedDirection) {
        lightDirection = normalize(lightDirection);
        lightDistance = dot(lightSample.p - lastVertex.p, lightDirection);
        if (baseShadow.lightIdx < params.num_lights) {
            auto& light = params.lights[baseShadow.lightIdx];
            if (light.shape == zeno::LightShape::Plane ||
                light.shape == zeno::LightShape::Ellipse)
            {
                // A finite directional emitter constrains both the outgoing
                // direction and its footprint. Moving the last caster vertex
                // can move the light endpoint outside the original shape.
                LightSampleRecord endpoint;
                if (!light.rect.hitAsLight(
                        &endpoint, lastVertex.p + params.cam.eye, lightDirection))
                {
                    illumination = {};
                    return true;
                }
                lightDistance = endpoint.dist;
            }
        }
        if (!(lightDistance > MNEE_MIN_DISTANCE)) {
            illumination = {};
            return true;
        }
    }
    else if (!normalizeWithLength(lightDirection, lightDistance, false)) {
        illumination = {};
        return true;
    }

    if (!lightFixedDirection) {
        float lightCosine = dot(lightSample.n, -lightDirection);
        if (lightDoubleSided) {
            lightCosine = fabsf(lightCosine);
        }
        if (!(lightCosine > 0.0f)) {
            illumination = {};
            return true;
        }
    }

    // The last caster-to-light segment uses the renderer's regular shadow test.
    // Glass absorption remains separate from surface shadow attenuation.
    ShadowPRD shadow = baseShadow;
    if (mediumCount > 0) {
        illumination *= float3(DisneyBSDF::Transmission(mediumExtinction[mediumCount - 1], lightDistance));
    }
    shadow.origin = lastVertex.p + lightDirection * MNEE_MIN_PROGRESS_DISTANCE;
    shadow.attanuation = make_float3(1.0f);
    shadow.nonThinTransHit = 0;
    traceOcclusion(params.handle,
                   shadow.origin,
                   lightDirection,
                   0.0f,
                   lightInfinite ? FLT_MAX : fmaxf(0.0f, lightDistance - MNEE_MIN_PROGRESS_DISTANCE),
                   &shadow,
                   ~LightMatMask & EverythingMask);
    illumination *= shadow.attanuation;
    return true;
}

static __forceinline__ __device__ bool kernel_path_mnee_sample(
    const float3& receiverPosition,
    const LightSampleRecord& lightSample,
    const float3& lightEmission,
    bool lightFixedDirection,
    bool lightDoubleSided,
    uint32_t seed,
    const ShadowPRD& baseShadow,
    float3& receiverDirection,
    float3& illumination)
{
    const float3 seedDirection = lightSample.dir;
    const bool lightInfinite = lightSample.dist == FLT_MAX;
    const float seedTmax = lightInfinite ? FLT_MAX : lightSample.dist;
    if (length(seedDirection) <= 0.0f || (seedTmax <= MNEE_MIN_DISTANCE)) {
        return false;
    }

    // 1. Seed the interface sequence along the original shadow ray. Occluders
    // are skipped here; only the final refracted chain decides visibility.
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS];
    int vertexCount = 0;
    float tmin = MNEE_MIN_PROGRESS_DISTANCE;
    for (int intersectionIndex = 0; intersectionIndex < MNEE_MAX_INTERSECTION_COUNT && tmin < seedTmax; ++intersectionIndex)
    {
        Hit hit;
        if (!traceProbe(receiverPosition, seedDirection, tmin, seedTmax, hit)) {
            break;
        }
        if (hit.sbtData != nullptr && hit.sbtData->causticCaster) {
            if (vertexCount >= MNEE_MAX_CAUSTIC_CASTERS || !mnee_setup_manifold_vertex(hit, seedDirection, seed, vertices[vertexCount]))
            {
                return false;
            }
            ++vertexCount;
        }
        tmin = advanceT(hit.t);
    }
    if (vertexCount == 0) {
        return false;
    }

    // 2. Solve Snell constraints using OptiX surface projection (no CH invocation).
    if (!mnee_newton_solver(receiverPosition,
                     baseShadow.ShadowNormal,
                     lightSample,
                     lightFixedDirection,
                     seed,
                     vertexCount,
                     vertices,
                     [](const float3& origin, const float3& target, uint32_t instanceId,
                        const ManifoldMaterial& material, uint32_t seed, ManifoldGeometry& projected) {
                         return projectToInstance(origin, target, instanceId, material, seed, projected);
                     }))
    {
        return false;
    }

    // 3. Evaluate the converged path, including transmission tint and visibility.
    return mnee_path_contribution(receiverPosition, lightSample, lightEmission,
                                  lightFixedDirection, lightDoubleSided, seed, baseShadow,
                                  receiverDirection, illumination, vertexCount, vertices);
}

} // namespace Mnee
