#pragma once
#include <stdint.h>
#include <string>
#include "patch_info.h"

namespace zeno::rxmesh {
/**
 * @brief A unique handle for each vertex/edge/face. Can be used to access mesh attributes
 */
struct ElementHandle {
    using LocalT = LocalIndexT;

    __device__ __host__ ElementHandle() : m_handle(INVALID64) {}

    __device__ __host__ ElementHandle(uint32_t     patch_id,
                                     LocalIndexT vertex_local_id) {
        uint64_t ret = patch_id;
        ret          = (ret << 32);
        ret |= vertex_local_id.id;
        m_handle = ret;
    }

    bool __device__ __host__ __inline__ operator==(
        const ElementHandle& rhs) const {
        return m_handle == rhs.m_handle;
    }
    bool __device__ __host__ __inline__ operator!=(
        const ElementHandle& rhs) const {
        return !(*this == rhs);
    }

    bool __device__ __host__ __inline__ is_valid() const {
        return m_handle != INVALID64;
    }

    uint64_t __device__ __host__ __inline__ unique_id() const {
        return m_handle;
    }

    std::pair<uint32_t, uint16_t> __device__ __host__ __inline__ unpack() const {
        uint16_t local_id = m_handle & ((1 << 16) - 1);
        uint32_t patch_id = m_handle >> 32;
        return std::make_pair(patch_id, local_id);
    }

   private:
    uint64_t m_handle;
};
}  // namespace rxmesh