#include "VDBDensityBake.h"

#include "NvrtcWorker.h"
#include "Octree.h"
#include "optixPathTracer.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace xinxinoptix {
namespace {

static constexpr uint32_t kDenseGpuOctreeStagingDepthLimit = 8u;

const char* fallbackDensityBakeSource()
{
    return "#include \"CallableVolume.cu\"\n";
}

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

struct DensityBakeModule {
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUfunction sparse_brick_kernel = nullptr;
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

std::string currentComputeArchitectureOption()
{
    int device = 0;
    cudaDeviceProp prop = {};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return {};
    }

    return "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
}

std::string makeBakeModuleKey(
    const std::string& source,
    const std::vector<std::string>& compile_macros,
    const std::string& architecture_option)
{
    std::string key = architecture_option;
    key.push_back('\n');
    for (const auto& macro : compile_macros) {
        key += macro;
        key.push_back('\n');
    }
    key += source;
    return key;
}

DensityBakeModule* compileDensityBakeModule(
    const std::string& source,
    const std::vector<std::string>& compile_macros,
    const std::string& architecture_option)
{
    auto module = std::make_unique<DensityBakeModule>();

    std::vector<const char*> compiler_options {
        "-std=c++17",
        "-default-device",
        "--use_fast_math",
        "--zeno-nvrtc-worker=1",
        "--define-macro=__VDB_DENSITY_BAKE__",
    };

    if (!architecture_option.empty()) {
        compiler_options.push_back(architecture_option.c_str());
    }
    for (const auto& macro : compile_macros) {
        compiler_options.push_back(macro.c_str());
    }

    auto compile_result = zeno::nvrtc_worker::compile(
        source.c_str(),
        "",
        "CallableVolumeBake.cu",
        compiler_options);

    if (!compile_result.success) {
        std::cerr << "Failed to compile VDB density bake kernel";
        if (!compile_result.log.empty()) {
            std::cerr << ":\n" << compile_result.log;
        }
        std::cerr << std::endl;
        return nullptr;
    }

    if (!checkCudaDriver(cuInit(0), "cuInit")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleLoadData(&module->module, compile_result.data.data()), "cuModuleLoadData")) {
        return nullptr;
    }

    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_brick_kernel, module->module, "bakeNanoVDBDensityToSparseBricks"), "cuModuleGetFunction(bakeNanoVDBDensityToSparseBricks)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_init_kernel, module->module, "initBakedSparseVolumeBuffers"), "cuModuleGetFunction(initBakedSparseVolumeBuffers)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_accumulate_octree_kernel, module->module, "accumulateBakedSparseVolumeOctreeLeaves"), "cuModuleGetFunction(accumulateBakedSparseVolumeOctreeLeaves)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_reduce_octree_kernel, module->module, "reduceBakedSparseVolumeOctreeLevel"), "cuModuleGetFunction(reduceBakedSparseVolumeOctreeLevel)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_count_compact_children_kernel, module->module, "countBakedSparseVolumeCompactChildren"), "cuModuleGetFunction(countBakedSparseVolumeCompactChildren)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_prefix_compact_children_kernel, module->module, "prefixBakedSparseVolumeCompactChildren"), "cuModuleGetFunction(prefixBakedSparseVolumeCompactChildren)")) {
        return nullptr;
    }
    if (!checkCudaDriver(cuModuleGetFunction(&module->sparse_emit_compact_level_kernel, module->module, "emitBakedSparseVolumeCompactLevel"), "cuModuleGetFunction(emitBakedSparseVolumeCompactLevel)")) {
        return nullptr;
    }

    if (!checkCudaDriver(cuModuleGetGlobal(&module->params_symbol, &module->params_symbol_size, module->module, "params"), "cuModuleGetGlobal(params)")) {
        return nullptr;
    }

    return module.release();
}

DensityBakeModule* ensureDensityBakeKernel(const VDBDensityBakeInputs& inputs)
{
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<DensityBakeModule>> cache;

    std::string source = inputs.callable_source != nullptr
        ? std::string(inputs.callable_source)
        : std::string(fallbackDensityBakeSource());
    std::string architecture_option = currentComputeArchitectureOption();
    const std::string key = makeBakeModuleKey(source, inputs.compile_macros, architecture_option);

    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second.get();
    }

    auto* module = compileDensityBakeModule(source, inputs.compile_macros, architecture_option);
    if (module == nullptr) {
        return nullptr;
    }

    auto [it, _] = cache.emplace(key, std::unique_ptr<DensityBakeModule>(module));
    return it->second.get();
}

Params makeBakeParams(const Params& params)
{
    Params bake_params = params;
    bake_params.cam.eye.x = 0.0f;
    bake_params.cam.eye.y = 0.0f;
    bake_params.cam.eye.z = 0.0f;
    return bake_params;
}

uint32_t denseOctreeLevelStart(uint32_t level)
{
    return ((1u << (3u * level)) - 1u) / 7u;
}

uint32_t denseOctreeNodeCount(uint32_t octreeBuildDepth)
{
    return denseOctreeLevelStart(octreeBuildDepth + 1u);
}

} // namespace

bool bakeNanoVDBGridToSparseBricks(
    std::size_t grid_size,
    BakedSparseVolumeDevice& device_volume,
    const VDBDensityBakeInputs& inputs,
    const VDBDensityBakeOptions& options,
    VDBDensityBakeResult* result)
{
    if (grid_size == 0 || device_volume.brick_count == 0 ||
        device_volume.brick_table == nullptr || device_volume.brick_origins == nullptr ||
        device_volume.voxel_values == nullptr) {
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

    uint32_t octreeBuildDepth = bakedSparseVolumeClampOctreeBuildDepth(device_volume.octreeBuildDepth);
    const bool buildDenseGpuOctree = device_volume.octree != nullptr && octreeBuildDepth <= kDenseGpuOctreeStagingDepthLimit;
    const uint32_t bottom_count = buildDenseGpuOctree ? (1u << (3u * octreeBuildDepth)) : 0u;
    const uint32_t dense_octree_node_count = buildDenseGpuOctree ? denseOctreeNodeCount(octreeBuildDepth) : 0u;
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
    };

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
        &max_density_bits,
    };
    const unsigned int bake_blocks = std::min<unsigned int>(std::max<uint32_t>(device_volume.brick_count, 1u), 65535u);
    CudaEventScope density_start_event;
    CudaEventScope density_end_event;
    const auto sparse_wall_start = std::chrono::steady_clock::now();
    if (density_start_event.valid()) {
        cuEventRecord(density_start_event.event, nullptr);
    }
    if (!checkCudaDriver(cuLaunchKernel(
            bake_module->sparse_brick_kernel,
            bake_blocks, 1, 1,
            block_size, 1, 1,
            0, nullptr,
            bake_args, nullptr), "cuLaunchKernel(bakeNanoVDBDensityToSparseBricks)")) {
        cleanup();
        return false;
    }
    if (density_end_event.valid()) {
        cuEventRecord(density_end_event.event, nullptr);
    }
    if (!checkCudaDriver(cuStreamSynchronize(nullptr), "cuStreamSynchronize(bakeNanoVDBDensityToSparseBricks)")) {
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
    if (!checkCudaDriver(cuLaunchKernel(
            bake_module->sparse_accumulate_octree_kernel,
            bake_blocks, 1, 1,
            block_size, 1, 1,
            0, nullptr,
            accum_args, nullptr), "cuLaunchKernel(accumulateBakedSparseVolumeOctreeLeaves)")) {
        cleanup();
        return false;
    }
    if (octree_accumulate_event.valid()) {
        cuEventRecord(octree_accumulate_event.event, nullptr);
    }

    for (int level = int(octreeBuildDepth); level >= 0; --level) {
        const uint32_t node_count = 1u << (3u * uint32_t(level));
        const unsigned int reduce_blocks = (node_count + block_size - 1u) / block_size;
        uint32_t level_arg = uint32_t(level);
        void* reduce_args[] = {
            &device_volume,
            &leaf_min_bits,
            &leaf_max_bits,
            &leaf_coverage,
            &leaf_quantized_sum,
            &level_arg,
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

    for (uint32_t level = 0; level <= octreeBuildDepth && current_count != 0u; ++level) {
        const unsigned int compact_blocks = (current_count + block_size - 1u) / block_size;

        if (level < octreeBuildDepth) {
            uint32_t dense_parent_count = 1u << (3u * level);
            const unsigned int dense_parent_blocks = (dense_parent_count + block_size - 1u) / block_size;
            void* count_args[] = {
                &dense_octree,
                &current_dense_indices,
                &dense_parent_count,
                &level,
                &octreeBuildDepth,
                &child_counts,
            };
            if (!checkCudaDriver(cuLaunchKernel(
                    bake_module->sparse_count_compact_children_kernel,
                    dense_parent_blocks, 1, 1,
                    block_size, 1, 1,
                    0, nullptr,
                    count_args, nullptr), "cuLaunchKernel(countBakedSparseVolumeCompactChildren)")) {
                cleanup();
                return false;
            }

            void* prefix_args[] = {
                &child_counts,
                &child_offsets,
                &dense_parent_count,
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
                &next_dense_indices,
                &child_offsets,
                &current_count,
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
                &next_dense_indices,
                &child_offsets,
                &current_count,
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
        if (!checkCudaDriver(cuMemcpyDtoH(&max_density_host, max_density_bits, sizeof(max_density_host)), "cuMemcpyDtoH(maxDensityBits)")) {
            cleanup();
            return false;
        }
        std::memcpy(&result->max_density, &max_density_host, sizeof(result->max_density));
        result->sparse_total_wall_ms = std::chrono::duration<float, std::milli>(octree_wall_end - sparse_wall_start).count();
        result->sparse_density_gpu_ms = elapsedMs(density_start_event, density_end_event);
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
