#pragma once

#include <cuda.h>

#include <cstddef>
#include <string>
#include <vector>

#include "Octree.h"

struct HitGroupData;
struct Params;

namespace xinxinoptix {

enum class VolumeDensityBakeGridType {
    Unsupported, Float,
    Fp16, Fp8, Fp4
};

struct VolumeDensityBakeOptions {
    bool clamp_negative = true;
    bool validate_sparse_octree = false;
    int topology_padding_voxels = 0;
    uint8_t octreeBuildDepth = BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;
    bool use_custom_sample_bbox = false;
    int3 custom_sample_min {};
    int3 custom_sample_max {};
};

struct VolumeDensityBakeInputs {
    const Params* params = nullptr;
    const HitGroupData* hit_group = nullptr;
    const std::string* callable_ptx = nullptr;
    const std::string* callable_module_key = nullptr;
    const char* callable_source = nullptr;
    const char* density_signature = nullptr;
    const char* validation_label = nullptr;
    uint32_t seed = 1u;
};

void prepareVolumeDensityBakeModuleAsync(
    const std::string& callable_module_key,
    const std::string& callable_ptx);

void resetVolumeDensityModuleBuildStats();
void consumeVolumeDensityModuleBuildStats(unsigned int& cuda_module_count, uint64_t& cuda_module_time_ms);

struct VolumeDensityBakeResult {
    float max_density = 0.0f;
    float sparse_total_wall_ms = 0.0f;
    float sparse_density_gpu_ms = 0.0f;
    float sparse_bounds_gpu_ms = 0.0f;
    float sparse_octree_wall_ms = 0.0f;
    float sparse_octree_accumulate_ms = 0.0f;
    float sparse_octree_reduce_ms = 0.0f;
    float sparse_octree_compact_ms = 0.0f;
    float sparse_octree_gpu_ms = 0.0f;
};

bool bakeDensityToSparseBricks(
    std::size_t grid_size,
    BakedSparseVolumeDevice& device_volume,
    const VolumeDensityBakeInputs& inputs,
    const VolumeDensityBakeOptions& options,
    VolumeDensityBakeResult* result = nullptr);

} // namespace xinxinoptix
