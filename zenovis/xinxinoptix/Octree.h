#pragma once

#ifndef __CUDACC_RTC__

#include <openvdb/openvdb.h>
#include <cassert>
#include <cstdint>
#include <vector>
#include "Host.h"

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

    inline void setChildMask(uint8_t mask) {
        data |= uint32_t(mask) << 24;
    }

    inline void setChildOffset(uint32_t offset) {
        assert(offset <= 0x00FFFFFF);
        data = (data & 0xFF000000) | offset;
    }
};

static constexpr int OCTREE_DEPTH = 6;

#ifndef __CUDACC_RTC__

struct VolumeAggregate {
    using Vec3F = openvdb::Vec3f;
    using Box3F = openvdb::math::BBox<Vec3F>;
    
    Box3F octbox;
    std::vector<OcNode> octree;

    void aggregate(openvdb::FloatGrid& vgrid, float T = 1.0f);
};

#endif