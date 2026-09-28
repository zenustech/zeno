#pragma once

#include <cuda/cstdint.h>

enum class SharcPass : uint32_t
{
    Disabled = 0u,
    Update = 1u,
    Query = 2u,
};

// Plain storage shared by the host Params upload and NVRTC device code.
// Keep this type free of default member initializers: Params is an OptiX
// constant-memory launch parameter block.
struct SharcLaunchParameters
{
    uint64_t hashEntries;
    uint64_t accumulation;
    uint64_t resolved;
    uint32_t capacity;
    uint32_t frameIndex;
    uint32_t pass;
    uint32_t updateSpacing;
    float sceneScale;
    float radianceScale;
    float roughnessMin;
    uint32_t debugMode;
    uint32_t materialDemodulation;
    uint32_t shEncoding;
};

static_assert(sizeof(SharcLaunchParameters) == 64u,
              "Unexpected SHARC launch parameter layout");
