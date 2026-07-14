#pragma once
#include <vector_types.h>

static constexpr unsigned int BAKED_SPARSE_VOLUME_OCTREE_FORMAT_COMPACT = 1u;
static constexpr unsigned int BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH = 8u;
static constexpr unsigned int BAKED_SPARSE_VOLUME_MAX_OCTREE_DEPTH = 8u;
static constexpr unsigned int BAKED_SPARSE_VOLUME_BRICK_SIZE = 8u;

static constexpr int OCTREE_DEPTH = int(BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH);

#ifndef __CUDACC_RTC__

#include <openvdb/openvdb.h>
#include <cassert>
#include <cstdint>
#include <vector>
#include "Host.h"

#include <algorithm>

inline constexpr unsigned int bakedSparseVolumeClampOctreeBuildDepth(unsigned int depth)
{
    depth = std::min(depth, BAKED_SPARSE_VOLUME_MAX_OCTREE_DEPTH);
    return std::max(depth, 1u);
}
#endif

struct OcNode {
    uint32_t data = 0;
    __half min_d = 0;
    __half max_d = 0;

    inline uint8_t childMask() const {
        return uint8_t(data >> 24);
    }

    inline uint8_t& childMaskRef() const {
        return *((uint8_t*)&data + 3);
    }

    inline uint32_t childOffset() const {
        return data & 0x00FFFFFF;
    }

    inline uint16_t leafAverageBits() const {
        return uint16_t(data & 0x0000FFFF);
    }

    inline void setChildMask(uint8_t mask) {
        data |= uint32_t(mask) << 24;
    }

    inline void setChildOffset(uint32_t offset) {
        assert(offset <= 0x00FFFFFF);
        data = (data & 0xFF000000) | offset;
    }

    inline void setLeafAverageBits(uint16_t bits) {
        data = (data & 0xFF000000) | uint32_t(bits);
    }
};

struct BakedSparseVolumeDevice {
    // Padded octree domain. This can be larger than the logical sampling bbox.
    int3 voxel_min {};
    int3 voxel_max {};
    int3 voxel_dim {};
    // Logical bake/sample domain, exclusive max. Voxels outside this range bake to zero.
    int3 sample_min {};
    int3 sample_max {};
    int3 brick_dim {};
    unsigned int brick_size = BAKED_SPARSE_VOLUME_BRICK_SIZE;
    unsigned int brick_count = 0;
    unsigned int brick_table_count = 0;
    unsigned int octreeBuildDepth = BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;
    int* brick_table = nullptr;
    int3* brick_origins = nullptr;
    unsigned short* voxel_values = nullptr;
    unsigned short* brick_min = nullptr;
    unsigned short* brick_max = nullptr;
    OcNode* octree = nullptr;
    unsigned int octree_node_count = 0;
    unsigned int octree_format = 0;
};

#ifndef __CUDACC_RTC__

struct VolumeAggregate {
    using Vec3F = openvdb::Vec3f;
    using Box3F = openvdb::math::BBox<Vec3F>;
    
    Box3F octbox;
    std::vector<OcNode> octree;
    int octreeBuildDepth = OCTREE_DEPTH;

    void aggregate(openvdb::FloatGrid& vgrid, int buildDepth = OCTREE_DEPTH);
    void aggregate(openvdb::FloatGrid& vgrid, const openvdb::CoordBBox& bbox, int buildDepth = OCTREE_DEPTH);
};

#endif