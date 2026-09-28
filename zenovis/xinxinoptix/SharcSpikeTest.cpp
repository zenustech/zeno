#include "SharcResources.h"
#include "NvrtcWorker.h"

#include <cuda_runtime_api.h>
#include <sutil/sutil.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sutil {

std::vector<const char*>& getIncFileTab()
{
    static std::vector<const char*> files;
    return files;
}

std::vector<const char*>& getIncPathTab()
{
    static std::vector<const char*> paths;
    return paths;
}

const char* lookupIncFile(const char* name)
{
    const auto& paths = getIncPathTab();
    const auto found = std::find(paths.begin(), paths.end(), std::string_view(name));
    if (found == paths.end()) {
        throw std::runtime_error(std::string("Missing embedded NVRTC source: ") + name);
    }
    return getIncFileTab().at(static_cast<std::size_t>(found - paths.begin()));
}

} // namespace sutil

namespace {

struct alignas(16) HostAccumulationData {
    int32_t data[4];
    int32_t data_ext[4];
};

struct alignas(8) HostPackedData {
    uint16_t radiance[4];
    uint32_t radiance_data_ext;
    uint32_t sample_num_data;
    uint32_t sample_data;
    uint32_t sample_data_ext;
};

static_assert(sizeof(HostAccumulationData) == 32);
static_assert(sizeof(HostPackedData) == 24);

float halfBitsToFloat(uint16_t bits)
{
    const uint32_t sign = uint32_t(bits >> 15u);
    const uint32_t exponent = uint32_t(bits >> 10u) & 0x1fu;
    const uint32_t mantissa = uint32_t(bits) & 0x3ffu;

    float value = 0.0f;
    if (exponent == 0u) {
        value = std::ldexp(float(mantissa), -24);
    } else if (exponent < 31u) {
        value = std::ldexp(float(mantissa + 1024u), int(exponent) - 25);
    } else {
        value = mantissa == 0u
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    }
    return sign != 0u ? -value : value;
}

bool check(CUresult result, const char* operation)
{
    if (result == CUDA_SUCCESS) {
        return true;
    }
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    std::cerr << operation << " failed: "
              << (message != nullptr ? message : "unknown CUDA error") << '\n';
    return false;
}

bool nearlyEqual(float lhs, float rhs)
{
    return std::abs(lhs - rhs) <= 0.002f;
}

struct ApiProbe {
    CUmodule module = nullptr;
    CUfunction update = nullptr;
    CUfunction query = nullptr;

    ~ApiProbe()
    {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

bool compileApiProbe(ApiProbe& probe)
{
    static const char* source = R"(
#include "SharcCudaCompat.h"
#define SHARC_UPDATE 1
#define SHARC_QUERY 1
#define SHARC_ENABLE_GLSL 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 1
#define SHARC_ENABLE_FADE_ACCELERATION 0
#define SHARC_MATERIAL_DEMODULATION 0
#define SHARC_SEPARATE_EMISSIVE 0
#define SHARC_USE_FP16 0
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#include "SharcCommon.h"

static __forceinline__ __device__ SharcParameters zenoSharcParameters(
    uint capacity,
    HashGridKey* hashEntries,
    SharcAccumulationData* accumulation,
    SharcPackedData* resolved)
{
    SharcParameters parameters = {};
    parameters.hashGridParameters.cameraPosition = make_float3(0.0f);
    parameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    parameters.hashGridParameters.sceneScale = 50.0f;
    parameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    parameters.hashGridData.capacity = capacity;
    parameters.hashGridData.hashEntriesBuffer = hashEntries;
    parameters.radianceScale = 1000.0f;
    parameters.accumulationBuffer = accumulation;
    parameters.resolvedBuffer = resolved;
    return parameters;
}

static __forceinline__ __device__ SharcHitData zenoSharcTestHit()
{
    SharcHitData hit = {};
    hit.positionWorld = make_float3(0.0f, 0.0f, 1.0f);
    hit.normalWorld = make_float3(0.0f, 0.0f, 1.0f);
    hit.radianceDirectionWorld = make_float3(0.0f, 0.0f, 1.0f);
    hit.radianceDirectionWeight = 1.0f;
    return hit;
}

extern "C" __global__ void zenoSharcApiUpdate(
    uint capacity,
    HashGridKey* hashEntries,
    SharcAccumulationData* accumulation,
    SharcPackedData* resolved)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const SharcParameters parameters = zenoSharcParameters(
        capacity, hashEntries, accumulation, resolved);
    const SharcHitData hit = zenoSharcTestHit();

    SharcState state = {};
    SharcInit(state);
    const bool continueTracing = SharcUpdateHit(
        parameters,
        state,
        hit,
        make_float3(0.25f, 0.5f, 0.75f),
        0.5f);
    if (continueTracing) {
        SharcSetThroughput(state, make_float3(0.8f, 0.6f, 0.4f));
        SharcUpdateMiss(
            parameters, state, make_float3(0.5f, 0.25f, 0.125f));
    }
}

extern "C" __global__ void zenoSharcApiQuery(
    uint capacity,
    HashGridKey* hashEntries,
    SharcAccumulationData* accumulation,
    SharcPackedData* resolved,
    float4* queryResult)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const SharcParameters parameters = zenoSharcParameters(
        capacity, hashEntries, accumulation, resolved);
    const SharcHitData hit = zenoSharcTestHit();

    float3 cachedRadiance = make_float3(0.0f);
    const bool found = SharcGetCachedRadiance(
        parameters, hit, cachedRadiance, true);
    queryResult[0] = make_float4(
        cachedRadiance.x,
        cachedRadiance.y,
        cachedRadiance.z,
        found ? 1.0f : 0.0f);

    SharcHitData reverseHit = hit;
    reverseHit.radianceDirectionWorld = make_float3(0.0f, 0.0f, -1.0f);
    cachedRadiance = make_float3(0.0f);
    const bool reverseFound = SharcGetCachedRadiance(
        parameters, reverseHit, cachedRadiance, true);
    queryResult[1] = make_float4(
        cachedRadiance.x,
        cachedRadiance.y,
        cachedRadiance.z,
        reverseFound ? 1.0f : 0.0f);
}
)";

    int device = 0;
    cudaDeviceProp properties = {};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        std::cerr << "Unable to query CUDA architecture for SHARC API probe\n";
        return false;
    }
    const std::string architecture = "--gpu-architecture=compute_" +
        std::to_string(properties.major) + std::to_string(properties.minor);
    const std::vector<const char*> options {
        "-std=c++17",
        "-default-device",
        "--use_fast_math",
        "--zeno-nvrtc-worker=0",
        architecture.c_str(),
    };
    auto result = zeno::nvrtc_worker::compile(
        source, "", "SharcApiProbe.cu", options);
    if (!result.success) {
        std::cerr << "SHARC update/query API probe compilation failed:\n"
                  << result.log << '\n';
        return false;
    }

    if (!check(cuModuleLoadData(&probe.module, result.data.data()),
               "cuModuleLoadData(SHARC API probe)")) {
        return false;
    }
    return check(
               cuModuleGetFunction(
                   &probe.update, probe.module, "zenoSharcApiUpdate"),
               "cuModuleGetFunction(zenoSharcApiUpdate)") &&
        check(
               cuModuleGetFunction(
                   &probe.query, probe.module, "zenoSharcApiQuery"),
               "cuModuleGetFunction(zenoSharcApiQuery)");
}

} // namespace

int main()
{
    constexpr uint32_t entry_count = 32u;
    xinxinoptix::SharcResources resources;
    std::string error;
    if (!resources.initialize(entry_count, &error)) {
        std::cerr << error << '\n';
        return 1;
    }
    ApiProbe probe;
    if (!compileApiProbe(probe)) {
        return 10;
    }
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize(initial clear)")) {
        return 2;
    }

    uint32_t capacity = resources.entryCount();
    CUdeviceptr hash_entries_device = resources.hashEntries();
    CUdeviceptr accumulation_device = resources.accumulation();
    CUdeviceptr resolved_device = resources.resolved();
    void* update_arguments[] = {
        &capacity,
        &hash_entries_device,
        &accumulation_device,
        &resolved_device,
    };
    if (!check(
            cuLaunchKernel(
                probe.update,
                1u, 1u, 1u,
                1u, 1u, 1u,
                0u, nullptr,
                update_arguments, nullptr),
            "cuLaunchKernel(zenoSharcApiUpdate)") ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize(SHARC API update)")) {
        return 11;
    }

    xinxinoptix::SharcResolveSettings round_trip_settings;
    if (!resources.resolve(round_trip_settings, nullptr, &error) ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize(SHARC API resolve)")) {
        std::cerr << error << '\n';
        return 12;
    }

    CUdeviceptr query_result_device = 0u;
    if (!check(cuMemAlloc(&query_result_device, 2u * sizeof(float4)),
               "cuMemAlloc(SHARC query result)")) {
        return 13;
    }
    void* query_arguments[] = {
        &capacity,
        &hash_entries_device,
        &accumulation_device,
        &resolved_device,
        &query_result_device,
    };
    float4 query_result[2] = {};
    const bool query_succeeded = check(
            cuLaunchKernel(
                probe.query,
                1u, 1u, 1u,
                1u, 1u, 1u,
                0u, nullptr,
                query_arguments, nullptr),
            "cuLaunchKernel(zenoSharcApiQuery)") &&
        check(cuCtxSynchronize(), "cuCtxSynchronize(SHARC API query)") &&
        check(cuMemcpyDtoH(
                  query_result, query_result_device, sizeof(query_result)),
              "cuMemcpyDtoH(SHARC query result)");
    cuMemFree(query_result_device);
    if (!query_succeeded) {
        return 14;
    }
    if (query_result[0].w != 1.0f ||
        !nearlyEqual(query_result[0].x, 0.65f) ||
        !nearlyEqual(query_result[0].y, 0.65f) ||
        !nearlyEqual(query_result[0].z, 0.8f) ||
        query_result[1].w != 1.0f ||
        !nearlyEqual(query_result[1].x, 0.0f) ||
        !nearlyEqual(query_result[1].y, 0.0f) ||
        !nearlyEqual(query_result[1].z, 0.0f)) {
        std::cerr << "SHARC update/query round trip failed: found="
                  << query_result[0].w << ", radiance=("
                  << query_result[0].x << ", " << query_result[0].y << ", "
                  << query_result[0].z << "), reverse=("
                  << query_result[1].x << ", " << query_result[1].y << ", "
                  << query_result[1].z << ")\n";
        return 15;
    }
    if (!resources.clear(nullptr, &error) ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize(round-trip clear)")) {
        std::cerr << error << '\n';
        return 16;
    }

    std::vector<uint64_t> hash_entries(entry_count, 0u);
    std::vector<HostAccumulationData> accumulation(entry_count, {});
    hash_entries[0] = 1u;
    // SH L1 is zero for a diffuse sample. RGB (1, 2, 3) maps to
    // YCoCg (2, -2, 0), with one accumulated sample.
    accumulation[0].data[0] = 0;
    accumulation[0].data[1] = 0;
    accumulation[0].data[2] = 0;
    accumulation[0].data[3] = 2000;
    accumulation[0].data_ext[0] = -2000;
    accumulation[0].data_ext[1] = 0;
    accumulation[0].data_ext[2] = 1;

    if (!check(cuMemcpyHtoD(
            resources.hashEntries(), hash_entries.data(),
            hash_entries.size() * sizeof(hash_entries[0])),
            "cuMemcpyHtoD(hash entries)") ||
        !check(cuMemcpyHtoD(
            resources.accumulation(), accumulation.data(),
            accumulation.size() * sizeof(accumulation[0])),
            "cuMemcpyHtoD(accumulation)")) {
        return 3;
    }

    xinxinoptix::SharcResolveSettings settings;
    settings.radiance_scale = 1000.0f;
    settings.accumulation_frame_num = 32u;
    settings.stale_frame_num_max = 64u;
    if (!resources.resolve(settings, nullptr, &error)) {
        std::cerr << error << '\n';
        return 4;
    }
    if (!check(cuCtxSynchronize(), "cuCtxSynchronize(resolve)")) {
        return 5;
    }

    HostPackedData packed = {};
    HostAccumulationData cleared_accumulation = {};
    uint64_t retained_hash = 0u;
    if (!check(cuMemcpyDtoH(&packed, resources.resolved(), sizeof(packed)),
               "cuMemcpyDtoH(resolved)") ||
        !check(cuMemcpyDtoH(
            &cleared_accumulation, resources.accumulation(),
            sizeof(cleared_accumulation)),
            "cuMemcpyDtoH(accumulation)") ||
        !check(cuMemcpyDtoH(&retained_hash, resources.hashEntries(), sizeof(retained_hash)),
               "cuMemcpyDtoH(hash entry)")) {
        return 6;
    }

    const bool radiance_matches =
        nearlyEqual(halfBitsToFloat(packed.radiance[0]), 0.0f) &&
        nearlyEqual(halfBitsToFloat(packed.radiance[1]), 0.0f) &&
        nearlyEqual(halfBitsToFloat(packed.radiance[2]), 0.0f) &&
        nearlyEqual(halfBitsToFloat(packed.radiance[3]), 2.0f) &&
        nearlyEqual(halfBitsToFloat(
            static_cast<uint16_t>(packed.radiance_data_ext)), -2.0f) &&
        nearlyEqual(halfBitsToFloat(
            static_cast<uint16_t>(packed.radiance_data_ext >> 16u)), 0.0f) &&
        nearlyEqual(halfBitsToFloat(
            static_cast<uint16_t>(packed.sample_num_data)), 1.0f);
    const bool frame_metadata_matches =
        (packed.sample_data & 0xffffu) == 1u &&
        (packed.sample_data >> 16u) == 0u &&
        packed.sample_data_ext == 0u;
    const bool accumulation_cleared =
        cleared_accumulation.data[0] == 0u &&
        cleared_accumulation.data[1] == 0u &&
        cleared_accumulation.data[2] == 0u &&
        cleared_accumulation.data[3] == 0u &&
        cleared_accumulation.data_ext[0] == 0u &&
        cleared_accumulation.data_ext[1] == 0u &&
        cleared_accumulation.data_ext[2] == 0u &&
        cleared_accumulation.data_ext[3] == 0u;

    if (!radiance_matches || !frame_metadata_matches ||
        !accumulation_cleared || retained_hash != 1u) {
        std::cerr << "SHARC resolve verification failed: radiance=("
                  << halfBitsToFloat(packed.radiance[0]) << ", "
                  << halfBitsToFloat(packed.radiance[1]) << ", "
                  << halfBitsToFloat(packed.radiance[2]) << ", "
                  << halfBitsToFloat(packed.radiance[3]) << "), sample_data="
                  << packed.sample_data << ", hash=" << retained_hash << '\n';
        return 7;
    }

    if (!resources.clear(nullptr, &error) ||
        !check(cuCtxSynchronize(), "cuCtxSynchronize(final clear)")) {
        std::cerr << error << '\n';
        return 8;
    }
    uint64_t cleared_hash = 1u;
    if (!check(cuMemcpyDtoH(&cleared_hash, resources.hashEntries(), sizeof(cleared_hash)),
               "cuMemcpyDtoH(cleared hash entry)") ||
        cleared_hash != 0u) {
        std::cerr << "SHARC clear verification failed\n";
        return 9;
    }

    std::cout << "SHARC 1.8.3 CUDA update/resolve/query round trip passed\n";
    return 0;
}
