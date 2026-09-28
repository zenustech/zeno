#include "SharcCudaCompat.h"

#define SHARC_UPDATE 0
#define SHARC_QUERY 0
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 1
#define SHARC_ENABLE_FADE_ACCELERATION 0
#define SHARC_MATERIAL_DEMODULATION 0
#define SHARC_SEPARATE_EMISSIVE 0
#define SHARC_USE_FP16 0
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1

#include "SharcCommon.h"

static_assert(SHARC_VERSION_MAJOR == 1, "Unexpected SHARC major version");
static_assert(SHARC_VERSION_MINOR == 8, "Unexpected SHARC minor version");
static_assert(SHARC_VERSION_BUILD == 3, "Unexpected SHARC build version");
static_assert(sizeof(HashGridKey) == 8, "SHARC hash key layout mismatch");
static_assert(sizeof(SharcAccumulationData) == 32,
              "SHARC accumulation layout mismatch");
static_assert(sizeof(SharcPackedData) == 24,
              "SHARC resolved layout mismatch");

extern "C" __global__ void zenoSharcResolve(
    uint capacity,
    HashGridKey* hashEntries,
    SharcAccumulationData* accumulation,
    SharcPackedData* resolved,
    float cameraPositionX,
    float cameraPositionY,
    float cameraPositionZ,
    float cameraPositionPrevX,
    float cameraPositionPrevY,
    float cameraPositionPrevZ,
    float sceneScale,
    float radianceScale,
    uint accumulationFrameNum,
    uint responsiveFrameNum,
    uint staleFrameNumMax,
    uint frameIndex)
{
    const uint entryIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (entryIndex >= capacity) {
        return;
    }

    SharcParameters sharcParameters = {};
    sharcParameters.hashGridParameters.cameraPosition = make_float3(
        cameraPositionX, cameraPositionY, cameraPositionZ);
    sharcParameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    sharcParameters.hashGridParameters.sceneScale = sceneScale;
    sharcParameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    sharcParameters.hashGridData.capacity = capacity;
    sharcParameters.hashGridData.hashEntriesBuffer = hashEntries;
    sharcParameters.radianceScale = radianceScale;
    sharcParameters.accumulationBuffer = accumulation;
    sharcParameters.resolvedBuffer = resolved;

    SharcResolveParameters resolveParameters = {};
    resolveParameters.cameraPositionPrev = make_float3(
        cameraPositionPrevX, cameraPositionPrevY, cameraPositionPrevZ);
    resolveParameters.accumulationFrameNum = accumulationFrameNum;
    resolveParameters.responsiveFrameNum = responsiveFrameNum;
    resolveParameters.staleFrameNumMax = staleFrameNumMax;
    resolveParameters.frameIndex = frameIndex;

    SharcResolveEntry(entryIndex, sharcParameters, resolveParameters);
}
