#pragma once
#include <vector_types.h>

#ifndef uchar
using uchar = unsigned char;
static_assert(sizeof(uchar) == 1);
#endif

static constexpr uchar BAKED_SPARSE_VOLUME_OCTREE_FORMAT_COMPACT = 1u;
static constexpr uchar BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH = 8u;
static constexpr uchar BAKED_SPARSE_VOLUME_MAX_OCTREE_DEPTH = 8u;
static constexpr uchar BAKED_SPARSE_VOLUME_BRICK_SIZE = 8u;

static constexpr uchar OCTREE_DEPTH = BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;

#ifndef __CUDACC_RTC__

#include <openvdb/openvdb.h>
#include <cassert>
#include <cstdint>
#include <vector>
#include "Host.h"

#include <algorithm>

inline constexpr uint8_t bakedSparseVolumeClampOctreeBuildDepth(uint8_t depth)
{
    depth = std::min<uint8_t>(depth, BAKED_SPARSE_VOLUME_MAX_OCTREE_DEPTH);
    return std::max<uint8_t>(depth, 1u);
}
#endif

static constexpr uint32_t OCTREE_MAX_RELATIVE_OFFSET = 0x00FFFFFFu;

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

    inline uint32_t childRelativeOffset() const {
        return data & OCTREE_MAX_RELATIVE_OFFSET;
    }

    inline uint16_t leafAverageBits() const {
        return uint16_t(data & 0x0000FFFF);
    }

    inline void setChildMask(uint8_t mask) {
        data = (data & OCTREE_MAX_RELATIVE_OFFSET) | (uint32_t(mask) << 24);
    }

    inline void setChildRelativeOffset(uint32_t offset) {
        assert(offset <= OCTREE_MAX_RELATIVE_OFFSET);
        data = (data & 0xFF000000) | (offset & OCTREE_MAX_RELATIVE_OFFSET);
    }

    inline void setLeafAverageBits(uint16_t bits) {
        data = (data & 0xFF000000) | uint32_t(bits);
    }
};

__host__ __device__ inline int roundUpToMultiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

__host__ __device__ inline int3 roundUpToMultiple(int x, int y, int z, int multiple) {
    int3 paddedDim {
        roundUpToMultiple(x, multiple),
        roundUpToMultiple(y, multiple),
        roundUpToMultiple(z, multiple) };
    return paddedDim;
}

struct BakedSparseVolumeDevice {
    // Padded octree domain. This can be larger than the logical sampling bbox.
    int3 voxel_min {};
    int3 voxel_max {};
    int3 voxel_dim {};
    // Logical bake/sample domain, exclusive max. Voxels outside this range bake to zero.
    int3 sample_min {};
    int3 sample_max {};
    int3 brick_dim {};
    unsigned int brick_count = 0;
    unsigned int brick_table_count = 0;
    unsigned int octree_node_count = 0;
    uint8_t brick_size = BAKED_SPARSE_VOLUME_BRICK_SIZE;
    uint8_t octreeBuildDepth = BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;
    uint8_t octree_format = 0;
    uint8_t reserved = 0;
    int* brick_table = nullptr;
    int3* brick_origins = nullptr;
    unsigned short* voxel_values = nullptr;
    unsigned short* brick_min = nullptr;
    unsigned short* brick_max = nullptr;
    OcNode* octree = nullptr;
};
static_assert(sizeof(BakedSparseVolumeDevice) == 136);

#ifndef __CUDACC_RTC__

struct VolumeAggregate {
    using Vec3F = openvdb::Vec3f;
    using Box3F = openvdb::math::BBox<Vec3F>;
    
    Box3F octbox;
    std::vector<OcNode> octree;
    uint8_t octreeBuildDepth = OCTREE_DEPTH;

    void aggregate(openvdb::FloatGrid& vgrid, uint8_t buildDepth = OCTREE_DEPTH);
    void aggregate(openvdb::FloatGrid& vgrid, const openvdb::CoordBBox& bbox, uint8_t buildDepth = OCTREE_DEPTH);
};

#endif