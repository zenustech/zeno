#pragma once
#include "../handle.h"

namespace zeno::rxmesh::detail {
template <typename LambdaT>
__global__ void for_each_vertex(const uint32_t   num_patches,
                                const PatchInfo* patch_info,
                                LambdaT          apply) {
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_v = patch_info[p_id].num_owned_vertices;
        for (uint16_t v = threadIdx.x; v < num_v; v += blockDim.x) { 
            const zeno::rxmesh::ElementHandle v_handle(p_id, v);
            apply(v_handle);
        }
    }
}


template <typename LambdaT>
__global__ void for_each_edge(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply) {
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_e = patch_info[p_id].num_owned_edges;
        for (uint16_t e = threadIdx.x; e < num_e; e += blockDim.x) {
            const zeno::rxmesh::ElementHandle e_handle(p_id, e);
            apply(e_handle);
        }
    }
}


template <typename LambdaT>
__global__ void for_each_face(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply) {
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_f = patch_info[p_id].num_owned_faces;
        for (uint16_t f = threadIdx.x; f < num_f; f += blockDim.x) {
            const zeno::rxmesh::ElementHandle f_handle(p_id, f);
            apply(f_handle);
        }
    }
}

}  // namespace rxmesh