#pragma once

namespace zeno::rxmesh {

template <typename T>
class Attribute;

template <typename T>
__global__ void memset_attribute(const Attribute<T> attr,
                                 const T            val,
                                 const uint16_t*    d_element_per_patch,
                                 const uint32_t     num_patches,
                                 const uint32_t     num_attributes) {
    uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t element_per_patch = d_element_per_patch[p_id];
        for (uint16_t i = threadIdx.x; i < element_per_patch; i += blockDim.x) {
            for (uint32_t j = 0; j < num_attributes; ++j) {
                attr(p_id, i, j) = val;
            }
        }
    }
}

}  // namespace rxmesh