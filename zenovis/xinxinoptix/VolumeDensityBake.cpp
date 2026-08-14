#include "VolumeDensityBake.h"

#include "Octree.h"
#include "optixPathTracer.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace xinxinoptix {
namespace {

static constexpr uint32_t kDenseGpuOctreeStagingDepthLimit = 8u;

const char* cudaDriverErrorString(CUresult result)
{
    const char* message = nullptr;
    if (cuGetErrorString(result, &message) == CUDA_SUCCESS && message != nullptr) {
        return message;
    }
    return "unknown CUDA driver error";
}

bool checkCudaDriver(CUresult result, const char* expr)
{
    if (result == CUDA_SUCCESS) {
        return true;
    }
    std::cerr << expr << " failed: " << cudaDriverErrorString(result) << std::endl;
    return false;
}

float halfBitsToFloat(uint16_t bits)
{
    const uint32_t sign = uint32_t(bits >> 15u);
    const uint32_t exponent = uint32_t(bits >> 10u) & 0x1Fu;
    const uint32_t mantissa = uint32_t(bits) & 0x03FFu;

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

bool checkCudaRuntime(cudaError_t result, const char* expr)
{
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << expr << " failed: " << cudaGetErrorString(result) << std::endl;
    return false;
}

struct DensityBakeModule {
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUfunction sparse_brick_kernel = nullptr;
    CUfunction sparse_cell_bounds_linear_kernel = nullptr;
    CUfunction sparse_cell_bounds_quadratic_kernel = nullptr;
    CUfunction sparse_cell_bounds_cubic_kernel = nullptr;
    CUfunction sparse_init_kernel = nullptr;
    CUfunction sparse_accumulate_octree_kernel = nullptr;
    CUfunction sparse_reduce_octree_kernel = nullptr;
    CUfunction sparse_count_compact_children_kernel = nullptr;
    CUfunction sparse_prefix_compact_children_kernel = nullptr;
    CUfunction sparse_emit_compact_level_kernel = nullptr;
    CUdeviceptr params_symbol = 0;
    std::size_t params_symbol_size = 0;

    ~DensityBakeModule()
    {
        if (module != nullptr) {
            cuModuleUnload(module);
        }
    }
};

struct CudaEventScope {
    CUevent event = nullptr;

    CudaEventScope()
    {
        cuEventCreate(&event, CU_EVENT_DEFAULT);
    }

    ~CudaEventScope()
    {
        if (event != nullptr) {
            cuEventDestroy(event);
        }
    }

    bool valid() const
    {
        return event != nullptr;
    }
};

float elapsedMs(const CudaEventScope& start, const CudaEventScope& end)
{
    if (!start.valid() || !end.valid()) {
        return 0.0f;
    }

    float ms = 0.0f;
    if (cuEventElapsedTime(&ms, start.event, end.event) != CUDA_SUCCESS) {
        return 0.0f;
    }
    return ms;
}

DensityBakeModule* loadDensityBakeModuleImage(const std::string& material_image)
{
    auto module = std::make_unique<DensityBakeModule>();

    if (!checkCudaDriver(cuInit(0), "cuInit")) {
        return nullptr;
    }
    if (!checkCudaDriver(
            cuModuleLoadData(&module->module, material_image.data()),
            "cuModuleLoadData(volume density bake)")) {
        return nullptr;
    }

    const auto load_function = [&](CUfunction& function, const char* name) {
        if (!checkCudaDriver(cuModuleGetFunction(&function, module->module, name), name)) {
            return false;
        }
        return checkCudaDriver(cuFuncLoad(function), name);
    };

    if (!load_function(module->sparse_brick_kernel, "bakeDensityToSparseBricks") ||
        !load_function(module->sparse_cell_bounds_linear_kernel, "bakeBakedSparseVolumeCellBoundsLinear") ||
        !load_function(module->sparse_cell_bounds_quadratic_kernel, "bakeBakedSparseVolumeCellBoundsQuadratic") ||
        !load_function(module->sparse_cell_bounds_cubic_kernel, "bakeBakedSparseVolumeCellBoundsCubic") ||
        !load_function(module->sparse_init_kernel, "initBakedSparseVolumeBuffers") ||
        !load_function(module->sparse_accumulate_octree_kernel, "accumulateBakedSparseVolumeOctreeLeaves") ||
        !load_function(module->sparse_reduce_octree_kernel, "reduceBakedSparseVolumeOctreeLevel") ||
        !load_function(module->sparse_count_compact_children_kernel, "countBakedSparseVolumeCompactChildren") ||
        !load_function(module->sparse_prefix_compact_children_kernel, "prefixBakedSparseVolumeCompactChildren") ||
        !load_function(module->sparse_emit_compact_level_kernel, "emitBakedSparseVolumeCompactLevel")) {
        return nullptr;
    }

    const CUresult params_result = cuModuleGetGlobal(
        &module->params_symbol,
        &module->params_symbol_size,
        module->module,
        "params");
    if (params_result == CUDA_ERROR_NOT_FOUND) {
        module->params_symbol = 0;
        module->params_symbol_size = 0;
    } else if (!checkCudaDriver(params_result, "cuModuleGetGlobal(params)")) {
        return nullptr;
    }

    return module.release();
}

using DensityBakeModulePtr = std::shared_ptr<DensityBakeModule>;
using DensityBakeModuleFuture = std::shared_future<DensityBakeModulePtr>;

struct DensityBakeModuleCache {
    std::mutex mutex;
    std::unordered_map<std::string, DensityBakeModuleFuture> entries;
};

DensityBakeModuleCache& densityBakeModuleCache()
{
    static DensityBakeModuleCache cache;
    return cache;
}

DensityBakeModuleFuture requestDensityBakeModule(
    const std::string& callable_module_key,
    const std::string& callable_ptx)
{
    if (callable_module_key.empty() || callable_ptx.empty()) {
        return {};
    }

    auto& cache = densityBakeModuleCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (auto it = cache.entries.find(callable_module_key); it != cache.entries.end()) {
        return it->second;
    }

    int device = 0;
    if (!checkCudaRuntime(cudaGetDevice(&device), "cudaGetDevice(volume density module)")) {
        return {};
    }

    auto future = std::async(
        std::launch::async,
        [callable_ptx, device]() -> DensityBakeModulePtr {
            if (!checkCudaRuntime(cudaSetDevice(device), "cudaSetDevice(volume density module)")) {
                return {};
            }
            try {
                return DensityBakeModulePtr(loadDensityBakeModuleImage(callable_ptx));
            } catch (const std::exception& error) {
                std::cerr << "volume density CUDA module preparation failed: "
                          << error.what() << std::endl;
                return {};
            }
        }).share();
    cache.entries.emplace(callable_module_key, future);
    return future;
}

void discardDensityBakeModule(const std::string& callable_module_key)
{
    auto& cache = densityBakeModuleCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.entries.erase(callable_module_key);
}

DensityBakeModule* ensureDensityBakeKernel(const VolumeDensityBakeInputs& inputs)
{
    if (inputs.callable_ptx == nullptr || inputs.callable_ptx->empty() ||
        inputs.callable_module_key == nullptr || inputs.callable_module_key->empty()) {
        std::cerr << "volume density bake requires compiled callable PTX" << std::endl;
        return nullptr;
    }

    auto future = requestDensityBakeModule(
        *inputs.callable_module_key,
        *inputs.callable_ptx);
    if (!future.valid()) {
        return nullptr;
    }

    auto module = future.get();
    if (module == nullptr) {
        discardDensityBakeModule(*inputs.callable_module_key);
        return nullptr;
    }
    return module.get();
}

Params makeBakeParams(const Params& params)
{
    Params bake_params = params;
    bake_params.cam.eye.x = 0.0f;
    bake_params.cam.eye.y = 0.0f;
    bake_params.cam.eye.z = 0.0f;
    return bake_params;
}

uint8_t densityBakeFilterOrder(const char* callable_source)
{
    if (callable_source == nullptr) {
        return 0u;
    }

    const std::string_view source(callable_source);
    if (source.find("samplingVDB<3,") != std::string_view::npos) return 3u;
    if (source.find("samplingVDB<2,") != std::string_view::npos) return 2u;
    if (source.find("samplingVDB<1,") != std::string_view::npos) return 1u;
    if (source.find("samplingVDB<4,") != std::string_view::npos) return 4u;
    return 0u;
}

uint32_t denseOctreeLevelStart(uint8_t level)
{
    return ((1u << (3u * level)) - 1u) / 7u;
}

uint32_t denseOctreeNodeCount(uint8_t octreeBuildDepth)
{
    return denseOctreeLevelStart(octreeBuildDepth + 1u);
}

} // namespace

void prepareVolumeDensityBakeModuleAsync(const std::string& callable_module_key, const std::string& callable_ptx)
{
    (void)requestDensityBakeModule(callable_module_key, callable_ptx);
}

bool bakeDensityToSparseBricks(
    std::size_t grid_size,
    BakedSparseVolumeDevice& device_volume,
    const VolumeDensityBakeInputs& inputs,
    const VolumeDensityBakeOptions& options,
    VolumeDensityBakeResult* result)
{
    if (grid_size == 0 || device_volume.brick_count == 0 || device_volume.brick_table == nullptr ||
        device_volume.brick_origins == nullptr || device_volume.voxel_values == nullptr) {
        return false;
    }

    const auto elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

const auto bake_module_begin = std::chrono::steady_clock::now();
    auto* bake_module = ensureDensityBakeKernel(inputs);
const auto bake_module_end = std::chrono::steady_clock::now();
std::cout << "\n bake cuda module cost:" << elapsed_ms(bake_module_begin, bake_module_end) << "ms \n";

    if (bake_module == nullptr) {
        return false;
    }

    Params fallback_params = {};
    HitGroupData fallback_hit_group = {};
    const Params& input_params = inputs.params != nullptr ? *inputs.params : fallback_params;
    HitGroupData hit_group = inputs.hit_group != nullptr ? *inputs.hit_group : fallback_hit_group;

    const Params bake_params = makeBakeParams(input_params);
    if (bake_module->params_symbol != 0 && bake_module->params_symbol_size >= sizeof(Params)) {
        if (!checkCudaDriver(cuMemcpyHtoD(bake_module->params_symbol, &bake_params, sizeof(Params)), "cuMemcpyHtoD(params)")) {
            return false;
        }
    }

    uint8_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(device_volume.octreeBuildDepth);
    const char* density_filter_source = inputs.density_signature != nullptr && inputs.density_signature[0] != '\0'
        ? inputs.density_signature
        : inputs.callable_source;
    device_volume.octree_filter = densityBakeFilterOrder(density_filter_source);
    const bool buildDenseGpuOctree = device_volume.octree != nullptr && octreeBuildDepth <= kDenseGpuOctreeStagingDepthLimit;
    const uint8_t filter_order = device_volume.octree_filter & BAKED_SPARSE_FILTER_ORDER_MASK;
    const bool requires_cell_bounds = !buildDenseGpuOctree;
    const uint32_t bottom_count = buildDenseGpuOctree ? (1u << (3u * octreeBuildDepth)) : 0u;
    const uint32_t dense_octree_node_count = buildDenseGpuOctree ? denseOctreeNodeCount(octreeBuildDepth) : 0u;
    const uint64_t cell_count = uint64_t(device_volume.brick_count) * 512ull;
    CUdeviceptr cell_min = 0;
    CUdeviceptr cell_max = 0;
    CUdeviceptr leaf_min_bits = 0;
    CUdeviceptr leaf_max_bits = 0;
    CUdeviceptr leaf_coverage = 0;
    CUdeviceptr leaf_quantized_sum = 0;
    CUdeviceptr compact_octree = 0;
    CUdeviceptr current_dense_indices = 0;
    CUdeviceptr next_dense_indices = 0;
    CUdeviceptr child_counts = 0;
    CUdeviceptr child_offsets = 0;
    CUdeviceptr level_counts = 0;
    CUdeviceptr max_density_bits = 0;
    unsigned int max_density_host = 0;

    auto cleanup = [&] {
        if (cell_min != 0) { cuMemFree(cell_min); }
        if (cell_max != 0) { cuMemFree(cell_max); }
        if (leaf_min_bits != 0) { cuMemFree(leaf_min_bits); }
        if (leaf_max_bits != 0) { cuMemFree(leaf_max_bits); }
        if (leaf_coverage != 0) { cuMemFree(leaf_coverage); }
        if (leaf_quantized_sum != 0) { cuMemFree(leaf_quantized_sum); }
        if (compact_octree != 0) { cuMemFree(compact_octree); }
        if (current_dense_indices != 0) { cuMemFree(current_dense_indices); }
        if (next_dense_indices != 0) { cuMemFree(next_dense_indices); }
        if (child_counts != 0) { cuMemFree(child_counts); }
        if (child_offsets != 0) { cuMemFree(child_offsets); }
        if (level_counts != 0) { cuMemFree(level_counts); }
        if (max_density_bits != 0) { cuMemFree(max_density_bits); }
        device_volume.cell_min = nullptr;
        device_volume.cell_max = nullptr;
    };

    if (requires_cell_bounds) {
        if (!checkCudaDriver(cuMemAlloc(&cell_min, sizeof(unsigned short) * cell_count), "cuMemAlloc(cellMin)") ||
            !checkCudaDriver(cuMemAlloc(&cell_max, sizeof(unsigned short) * cell_count), "cuMemAlloc(cellMax)")) {
            cleanup();
            return false;
        }
        device_volume.cell_min = reinterpret_cast<unsigned short*>(cell_min);
        device_volume.cell_max = reinterpret_cast<unsigned short*>(cell_max);
    }

    if (buildDenseGpuOctree &&
        (!checkCudaDriver(cuMemAlloc(&leaf_min_bits, sizeof(unsigned int) * bottom_count), "cuMemAlloc(leafMinBits)") ||
         !checkCudaDriver(cuMemAlloc(&leaf_max_bits, sizeof(unsigned int) * bottom_count), "cuMemAlloc(leafMaxBits)") ||
         !checkCudaDriver(cuMemAlloc(&leaf_coverage, sizeof(unsigned int) * bottom_count), "cuMemAlloc(leafCoverage)") ||
         !checkCudaDriver(cuMemAlloc(&leaf_quantized_sum, sizeof(unsigned long long) * bottom_count), "cuMemAlloc(leafQuantizedSum)") ||
         !checkCudaDriver(cuMemAlloc(&compact_octree, sizeof(OcNode) * dense_octree_node_count), "cuMemAlloc(compactOctree)") ||
         !checkCudaDriver(cuMemAlloc(&current_dense_indices, sizeof(uint32_t) * bottom_count), "cuMemAlloc(currentDenseIndices)") ||
         !checkCudaDriver(cuMemAlloc(&next_dense_indices, sizeof(uint32_t) * bottom_count), "cuMemAlloc(nextDenseIndices)") ||
         !checkCudaDriver(cuMemAlloc(&child_counts, sizeof(uint32_t) * bottom_count), "cuMemAlloc(childCounts)") ||
         !checkCudaDriver(cuMemAlloc(&child_offsets, sizeof(uint32_t) * bottom_count), "cuMemAlloc(childOffsets)") ||
         !checkCudaDriver(cuMemAlloc(&level_counts, sizeof(uint32_t) * (octreeBuildDepth + 1u)), "cuMemAlloc(levelCounts)"))) {
        cleanup();
        return false;
    }

    if (!checkCudaDriver(cuMemAlloc(&max_density_bits, sizeof(unsigned int)), "cuMemAlloc(maxDensityBits)")) {
        cleanup();
        return false;
    }

    if (!checkCudaDriver(cuMemsetD32(max_density_bits, 0, 1), "cuMemsetD32(maxDensityBits)") ||
        (buildDenseGpuOctree && (!checkCudaDriver(cuMemsetD32(level_counts, 0, octreeBuildDepth + 1u), "cuMemsetD32(levelCounts)") ||
                                 !checkCudaDriver(cuMemsetD32(level_counts, 1, 1), "cuMemsetD32(levelCounts[0])") ||
                                 !checkCudaDriver(cuMemsetD32(current_dense_indices, 0, 1), "cuMemsetD32(currentDenseIndices[0])")))) {
        cleanup();
        return false;
    }

    constexpr unsigned int block_size = 256;
    const uint32_t init_count = std::max<uint32_t>(device_volume.brick_table_count, bottom_count);
    const unsigned int init_blocks = std::min<unsigned int>((init_count + block_size - 1u) / block_size, 4096u);
    void* init_args[] = {
        &device_volume,
        &leaf_min_bits,
        &leaf_max_bits,
        &leaf_coverage,
        &leaf_quantized_sum,
    };
    if (!checkCudaDriver(cuLaunchKernel(
            bake_module->sparse_init_kernel,
            init_blocks, 1, 1,
            block_size, 1, 1,
            0, nullptr,
            init_args, nullptr), "cuLaunchKernel(initBakedSparseVolumeBuffers)")) {
        cleanup();
        return false;
    }

    int clamp_negative = options.clamp_negative ? 1 : 0;
    uint32_t seed = inputs.seed;
    void* bake_args[] = {
        &device_volume,
        &hit_group,
        &clamp_negative,
        &seed,
    };
    const unsigned int bake_blocks = std::min<unsigned int>(std::max<uint32_t>(device_volume.brick_count, 1u), 65535u);
    CudaEventScope density_start_event;
    CudaEventScope density_end_event;
    CudaEventScope bounds_end_event;
    const auto sparse_wall_start = std::chrono::steady_clock::now();
    if (density_start_event.valid()) {
        cuEventRecord(density_start_event.event, nullptr);
    }
    if (!checkCudaDriver(cuLaunchKernel(
            bake_module->sparse_brick_kernel,
            bake_blocks, 1, 1,
            block_size, 1, 1,
            0, nullptr,
            bake_args, nullptr), "cuLaunchKernel(bakeDensityToSparseBricks)")) {
        cleanup();
        return false;
    }
    if (density_end_event.valid()) {
        cuEventRecord(density_end_event.event, nullptr);
    }

    void* cell_bounds_args[] = {
        &device_volume,
        &max_density_bits,
    };
    CUfunction cell_bounds_kernel = bake_module->sparse_cell_bounds_linear_kernel;
    uint64_t cells_per_bounds_block = block_size;
    if (filter_order == 2u) {
        cell_bounds_kernel = bake_module->sparse_cell_bounds_quadratic_kernel;
        cells_per_bounds_block = 8ull;
    } else if (filter_order == 3u) {
        cell_bounds_kernel = bake_module->sparse_cell_bounds_cubic_kernel;
        cells_per_bounds_block = 8ull;
    }
    const unsigned int cell_bounds_blocks = static_cast<unsigned int>(std::min<uint64_t>(
        (cell_count + cells_per_bounds_block - 1ull) / cells_per_bounds_block,
        65535ull));
    if (requires_cell_bounds) {
        if (!checkCudaDriver(cuLaunchKernel(
                cell_bounds_kernel,
                cell_bounds_blocks, 1, 1,
                block_size, 1, 1,
                0, nullptr,
                cell_bounds_args, nullptr), "cuLaunchKernel(bakeBakedSparseVolumeCellBounds*)")) {
            cleanup();
            return false;
        }
    }
    if (bounds_end_event.valid()) {
        cuEventRecord(bounds_end_event.event, nullptr);
    }
    if (!checkCudaDriver(cuStreamSynchronize(nullptr), "cuStreamSynchronize(bakeDensityToSparseBricks)")) {
        cleanup();
        return false;
    }

    CudaEventScope octree_start_event;
    CudaEventScope octree_accumulate_event;
    CudaEventScope octree_reduce_event;
    CudaEventScope octree_compact_event;
    const auto octree_wall_start = std::chrono::steady_clock::now();

    if (!buildDenseGpuOctree) {
        if (!checkCudaDriver(cuMemcpyDtoH(&max_density_host, max_density_bits, sizeof(max_density_host)), "cuMemcpyDtoH(maxDensityBits)")) {
            cleanup();
            return false;
        }
        if (result != nullptr) {
            const auto sparse_wall_end = std::chrono::steady_clock::now();
            std::memcpy(&result->max_density, &max_density_host, sizeof(result->max_density));
            result->sparse_total_wall_ms = std::chrono::duration<float, std::milli>(sparse_wall_end - sparse_wall_start).count();
            result->sparse_density_gpu_ms = elapsedMs(density_start_event, density_end_event);
            result->sparse_bounds_gpu_ms = elapsedMs(density_end_event, bounds_end_event);
            result->sparse_octree_wall_ms = 0.0f;
            result->sparse_octree_accumulate_ms = 0.0f;
            result->sparse_octree_reduce_ms = 0.0f;
            result->sparse_octree_compact_ms = 0.0f;
            result->sparse_octree_gpu_ms = 0.0f;
        }
        device_volume.octree_node_count = 0u;
        device_volume.octree_format = BAKED_SPARSE_VOLUME_OCTREE_FORMAT_COMPACT;
        cleanup();
        return true;
    }

    if (octree_start_event.valid()) {
        cuEventRecord(octree_start_event.event, nullptr);
    }

    void* accum_args[] = {
        &device_volume,
        &leaf_min_bits,
        &leaf_max_bits,
        &leaf_coverage,
        &leaf_quantized_sum,
    };
    constexpr unsigned int accumulation_warps_per_block = block_size / 32u;
    const unsigned int accumulation_blocks = std::min<unsigned int>(
        (bottom_count + accumulation_warps_per_block - 1u) / accumulation_warps_per_block,
        65535u);
    if (!checkCudaDriver(cuLaunchKernel(
            bake_module->sparse_accumulate_octree_kernel,
            accumulation_blocks, 1, 1,
            block_size, 1, 1,
            0, nullptr,
            accum_args, nullptr), "cuLaunchKernel(accumulateBakedSparseVolumeOctreeLeaves)")) {
        cleanup();
        return false;
    }
    if (octree_accumulate_event.valid()) {
        cuEventRecord(octree_accumulate_event.event, nullptr);
    }

    for (int8_t level = octreeBuildDepth; level>=0; --level) {
        const uint32_t node_count = 1u << (3u * uint32_t(level));
        const unsigned int reduce_blocks = (node_count + block_size - 1u) / block_size;
        void* reduce_args[] = {
            &device_volume,
            &leaf_min_bits,
            &leaf_max_bits,
            &leaf_coverage,
            &leaf_quantized_sum,
            &level,
        };
        if (!checkCudaDriver(cuLaunchKernel(
                bake_module->sparse_reduce_octree_kernel,
                reduce_blocks, 1, 1,
                block_size, 1, 1,
                0, nullptr,
                reduce_args, nullptr), "cuLaunchKernel(reduceBakedSparseVolumeOctreeLevel)")) {
            cleanup();
            return false;
        }
    }
    if (octree_reduce_event.valid()) {
        cuEventRecord(octree_reduce_event.event, nullptr);
    }

    if (!checkCudaDriver(cuStreamSynchronize(nullptr), "cuStreamSynchronize")) {
        cleanup();
        return false;
    }

    uint32_t current_count = 1u;
    uint32_t compact_level_start = 0u;
    uint32_t compact_node_count = 0u;
    CUdeviceptr dense_octree = reinterpret_cast<CUdeviceptr>(device_volume.octree);

    for (uint8_t level = 0; level <= octreeBuildDepth && current_count != 0u; ++level) {
        const unsigned int compact_blocks = (current_count + block_size - 1u) / block_size;

        if (level < octreeBuildDepth) {
            void* count_args[] = {
                &dense_octree,
                &current_dense_indices,
                &current_count,
                &level,
                &octreeBuildDepth,
                &child_counts,
            };
            if (!checkCudaDriver(cuLaunchKernel(
                    bake_module->sparse_count_compact_children_kernel,
                    compact_blocks, 1, 1,
                    block_size, 1, 1,
                    0, nullptr,
                    count_args, nullptr), "cuLaunchKernel(countBakedSparseVolumeCompactChildren)")) {
                cleanup();
                return false;
            }

            void* prefix_args[] = {
                &child_counts,
                &child_offsets,
                &current_count,
                &level_counts,
                &level,
                &octreeBuildDepth,
            };
            if (!checkCudaDriver(cuLaunchKernel(
                    bake_module->sparse_prefix_compact_children_kernel,
                    1, 1, 1,
                    1, 1, 1,
                    0, nullptr,
                    prefix_args, nullptr), "cuLaunchKernel(prefixBakedSparseVolumeCompactChildren)")) {
                cleanup();
                return false;
            }

            uint32_t next_count = 0u;
            if (!checkCudaDriver(
                    cuMemcpyDtoH(&next_count, level_counts + sizeof(uint32_t) * (level + 1u), sizeof(next_count)),
                    "cuMemcpyDtoH(compactLevelCount)")) {
                cleanup();
                return false;
            }

            uint32_t compact_next_level_start = compact_level_start + current_count;
            void* emit_args[] = {
                &dense_octree,
                &compact_octree,
                &current_dense_indices,
                &current_count,
                &next_dense_indices,
                &child_offsets,
                &child_counts,
                &level,
                &octreeBuildDepth,
                &compact_level_start,
                &compact_next_level_start,
            };
            if (!checkCudaDriver(cuLaunchKernel(
                    bake_module->sparse_emit_compact_level_kernel,
                    compact_blocks, 1, 1,
                    block_size, 1, 1,
                    0, nullptr,
                    emit_args, nullptr), "cuLaunchKernel(emitBakedSparseVolumeCompactLevel)")) {
                cleanup();
                return false;
            }

            if (level == 0u && next_count == 0u) {
                compact_node_count = 0u;
                break;
            }

            compact_level_start = compact_next_level_start;
            current_count = next_count;
            std::swap(current_dense_indices, next_dense_indices);
        } else {
            uint32_t unused_next_level_start = 0u;
            void* emit_args[] = {
                &dense_octree,
                &compact_octree,
                &current_dense_indices,
                &current_count,
                &next_dense_indices,
                &child_offsets,
                &child_counts,
                &level,
                &octreeBuildDepth,
                &compact_level_start,
                &unused_next_level_start,
            };
            if (!checkCudaDriver(cuLaunchKernel(
                    bake_module->sparse_emit_compact_level_kernel,
                    compact_blocks, 1, 1,
                    block_size, 1, 1,
                    0, nullptr,
                    emit_args, nullptr), "cuLaunchKernel(emitBakedSparseVolumeCompactLevel leaf)")) {
                cleanup();
                return false;
            }
            compact_node_count = compact_level_start + current_count;
        }
    }

    if (compact_node_count == 0u) {
        device_volume.octree_node_count = 0;
        device_volume.octree_format = 0;
    } else {
        if (!checkCudaDriver(
                cuMemcpyDtoD(dense_octree, compact_octree, sizeof(OcNode) * compact_node_count),
                "cuMemcpyDtoD(compactOctree)")) {
            cleanup();
            return false;
        }
        device_volume.octree_node_count = compact_node_count;
        device_volume.octree_format = BAKED_SPARSE_VOLUME_OCTREE_FORMAT_COMPACT;
    }
    if (octree_compact_event.valid()) {
        cuEventRecord(octree_compact_event.event, nullptr);
    }

    if (!checkCudaDriver(cuStreamSynchronize(nullptr), "cuStreamSynchronize(compactOctree)")) {
        cleanup();
        return false;
    }
    const auto octree_wall_end = std::chrono::steady_clock::now();

    if (result != nullptr) {
        if (requires_cell_bounds) {
            if (!checkCudaDriver(cuMemcpyDtoH(&max_density_host, max_density_bits, sizeof(max_density_host)), "cuMemcpyDtoH(maxDensityBits)")) {
                cleanup();
                return false;
            }
            std::memcpy(&result->max_density, &max_density_host, sizeof(result->max_density));
        } else {
            OcNode root_node {};
            if (!checkCudaDriver(cuMemcpyDtoH(&root_node, dense_octree, sizeof(root_node)), "cuMemcpyDtoH(octreeRoot)")) {
                cleanup();
                return false;
            }
            uint16_t root_max_bits = 0u;
            std::memcpy(&root_max_bits, &root_node.max_d, sizeof(root_max_bits));
            result->max_density = halfBitsToFloat(root_max_bits);
        }
        result->sparse_total_wall_ms = std::chrono::duration<float, std::milli>(octree_wall_end - sparse_wall_start).count();
        result->sparse_density_gpu_ms = elapsedMs(density_start_event, density_end_event);
        result->sparse_bounds_gpu_ms = elapsedMs(density_end_event, bounds_end_event);
        result->sparse_octree_wall_ms = std::chrono::duration<float, std::milli>(octree_wall_end - octree_wall_start).count();
        result->sparse_octree_accumulate_ms = elapsedMs(octree_start_event, octree_accumulate_event);
        result->sparse_octree_reduce_ms = elapsedMs(octree_accumulate_event, octree_reduce_event);
        result->sparse_octree_compact_ms = elapsedMs(octree_reduce_event, octree_compact_event);
        result->sparse_octree_gpu_ms = elapsedMs(octree_start_event, octree_compact_event);
    }

    cleanup();
    return true;
}

} // namespace xinxinoptix
