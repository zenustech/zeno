#pragma once

#include <optix.h>
#include <optix_device.h>
#include <cuda/cstdint.h>

// An SBT pointer is opaque to hit reconstruction; shader policy belongs in Mnee.h.
struct HitGroupData;

namespace Mnee {
struct Hit {
    float t = 0.0f;
    float2 barys = {};
    uint32_t primitiveIndex = 0;
    uint32_t sbtGasIndex = 0;
    uint32_t instanceId = 0;
    uint32_t casterId = 0;
    uint32_t instanceIndex = 0;
    unsigned int hitKind = 0;
    OptixTraversableHandle gas = 0;
    HitGroupData* sbtData = nullptr;
    float4 objectToWorld[3];
    float4 worldToObject[3];
};

// These APIs inspect the outgoing hit produced by optixTraverse, not the
// incoming closest-hit state. Consume it before issuing another traversal.
// OptiX 9 supplies the GAS directly: the last transform need not be an instance
// whose immediate child is a GAS (static/motion transforms may intervene).
static __forceinline__ __device__ bool outgoingHitTransforms(
    float4 objectToWorld[3], float4 worldToObject[3], OptixTraversableHandle& gas)
{
    gas = optixHitObjectGetGASTraversableHandle();
    if (gas == 0) {
        return false;
    }
    optixHitObjectGetObjectToWorldTransformMatrix(reinterpret_cast<float*>(objectToWorld));
    optixHitObjectGetWorldToObjectTransformMatrix(reinterpret_cast<float*>(worldToObject));
    return true;
}

static __forceinline__ __device__ bool captureOutgoingHit(Hit& hit)
{
    hit = Hit{};
    if (!optixHitObjectIsHit()) {
        return false;
    }
    hit.t = optixHitObjectGetRayTmax();
    hit.hitKind = optixHitObjectGetHitKind();

    if (optixGetPrimitiveType(hit.hitKind) == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        hit.barys = optixHitObjectGetTriangleBarycentrics();
    }
    hit.primitiveIndex = optixHitObjectGetPrimitiveIndex();
    hit.sbtGasIndex = optixHitObjectGetSbtGASIndex();
    hit.instanceId = optixHitObjectGetInstanceId();
    hit.instanceIndex = optixHitObjectGetInstanceIndex();
    // Keep the renderer's reported ID for MatInput. It can identify the outer
    // instance, so it does not distinguish two casters under the same parent.
    // The solver uses the deepest INSTANCE's globally unique CPU-assigned ID.
    // Static/motion transforms can occur after that instance; skip those.
    hit.casterId = hit.instanceId;
    for (int i = static_cast<int>(optixHitObjectGetTransformListSize()) - 1; i >= 0; --i) {
        const auto transform = optixHitObjectGetTransformListHandle(i);
        if (optixGetTransformTypeFromHandle(transform) == OPTIX_TRANSFORM_TYPE_INSTANCE) {
            hit.casterId = optixGetInstanceIdFromHandle(transform);
            break;
        }
    }
    hit.sbtData = reinterpret_cast<HitGroupData*>(optixHitObjectGetSbtDataPointer());
    // Visibility-only probes do not reconstruct transforms or evaluate materials.
    return hit.sbtData != nullptr;
}
} // namespace Mnee
