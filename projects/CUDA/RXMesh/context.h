#pragma once
#include "patch_info.h"

namespace zeno::rxmesh {

/**
 * @brief context for the mesh parameters and pointers. Everything is allocated
 * on and managed by RXMesh. This class is meant to be a vehicle to copy various
 * parameters to the device kernels.
 */
class Context {
   public:
    friend class RXMesh;

    Context() : m_num_patches(nullptr), m_patches_info(nullptr) {}
    Context(const Context&) = default;

    __device__ __forceinline__ uint32_t get_num_patches() const {
        return *m_num_patches;
    }
    __device__ __forceinline__ PatchInfo* get_patches_info() const {
        return m_patches_info;
    }

    static __device__ __host__ __forceinline__ void
    unpack_edge_dir(const uint16_t edge_dir, uint16_t& edge, flag_t& dir) {
        dir  = (edge_dir & 1) != 0;
        edge = edge_dir >> 1;
    }

   private:
    void init(const uint32_t num_patches, PatchInfo* patches) {
        CUDA_ERROR(cudaMalloc((void**)&m_num_patches, sizeof(uint32_t)));
        CUDA_ERROR(cudaMemcpy(m_num_patches,
                              &num_patches,
                              sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        m_patches_info = patches;
    }

    void release() {
        CUDA_ERROR(cudaFree(m_num_patches));
    }


    uint32_t *m_num_patches;
    PatchInfo* m_patches_info;
};
}  // namespace rxmesh