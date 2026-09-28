#pragma once

#include "SharcLaunchParameters.h"

#include "SharcCudaCompat.h"

#define SHARC_UPDATE 1
#define SHARC_QUERY 1
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 1
#define SHARC_ENABLE_FADE_ACCELERATION 0
#define SHARC_MATERIAL_DEMODULATION 1
#define SHARC_SEPARATE_EMISSIVE 0
#define SHARC_USE_FP16 0
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1

// Keep SHARC's HLSL helper names from colliding with renderer shader helpers.
#define saturate sharcSaturate
#include "SharcCommon.h"
#undef saturate

static_assert(SHARC_VERSION_MAJOR == 1, "Unexpected SHARC major version");
static_assert(SHARC_VERSION_MINOR == 8, "Unexpected SHARC minor version");
static_assert(SHARC_VERSION_BUILD == 3, "Unexpected SHARC build version");
static_assert(sizeof(HashGridKey) == 8, "SHARC hash key layout mismatch");
static_assert(sizeof(SharcAccumulationData) == 32,
              "SHARC accumulation layout mismatch");
static_assert(sizeof(SharcPackedData) == 24,
              "SHARC resolved layout mismatch");

// The SHARC sources have already been preprocessed at this point. Do not let
// their HLSL compatibility macros alter the renderer source that includes us.
#undef float2
#undef float3
#undef float4
#undef int2
#undef int3
#undef int4
#undef uint3
#undef uint4
#undef float16_t4
#undef RW_STRUCTURED_BUFFER
#undef BUFFER_AT_OFFSET
#undef HASH_GRID_CONST
#undef HASH_GRID_LOOP_ATTR
#undef InterlockedAdd
#undef InterlockedExchange
#undef InterlockedCompareExchange

static __forceinline__ __device__ bool zenoSharcIsPass(const SharcLaunchParameters& launch, SharcPass pass)
{
    return launch.pass == static_cast<uint32_t>(pass) &&
        launch.capacity != 0u && launch.hashEntries != 0u &&
        launch.accumulation != 0u && launch.resolved != 0u;
}

// SHARC accumulates into unsigned fixed-point storage. Keep renderer-side
// negative/NaN/Inf values from becoming persistent high-energy cache data.
static __forceinline__ __device__ float3 zenoSharcSanitizeSignal(const float3& value)
{
    return make_float3(
        isfinite(value.x) ? fmaxf(value.x, 0.0f) : 0.0f,
        isfinite(value.y) ? fmaxf(value.y, 0.0f) : 0.0f,
        isfinite(value.z) ? fmaxf(value.z, 0.0f) : 0.0f);
}

static __forceinline__ __device__ SharcParameters zenoSharcBuildParameters(const SharcLaunchParameters& launch, const float3& cameraPosition)
{
    SharcParameters parameters = {};
    parameters.hashGridParameters.cameraPosition = cameraPosition;
    parameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    parameters.hashGridParameters.sceneScale = launch.sceneScale;
    parameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    parameters.hashGridData.capacity = launch.capacity;
    parameters.hashGridData.hashEntriesBuffer = reinterpret_cast<HashGridKey*>(launch.hashEntries);
    parameters.radianceScale = launch.radianceScale;
    parameters.accumulationBuffer = reinterpret_cast<SharcAccumulationData*>(launch.accumulation);
    parameters.resolvedBuffer = reinterpret_cast<SharcPackedData*>(launch.resolved);
    return parameters;
}

static __forceinline__ __device__ SharcHitData zenoSharcHitData(
    const float3& positionWorld,
    const float3& geometryNormalWorld,
    const float3& materialDemodulation = make_float3(1.0f),
    const float3& radianceDirectionWorld = make_float3(0.0f, 0.0f, 1.0f),
    float radianceDirectionWeight = 0.0f)
{
    SharcHitData hit = {};
    hit.positionWorld = positionWorld;
    hit.normalWorld = geometryNormalWorld;
    hit.radianceDirectionWorld = radianceDirectionWorld;
    hit.radianceDirectionWeight = radianceDirectionWeight;
    hit.materialDemodulation = materialDemodulation;
    return hit;
}
