#pragma once
#include <assert.h>
#include "../utils/macros.h"

namespace zeno::rxmesh {
__global__ static void shift(const uint32_t num_faces,
                             uint32_t*      face_patch,                             
                             uint32_t*      patches_val) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_faces) {
        face_patch[tid] = face_patch[tid] >> 1;
        patches_val[tid] = patches_val[tid] >> 1;
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief Get the adjacent faces of a face
 */
__device__ __forceinline__ const uint32_t* get_face_faces(
    const uint32_t* d_ff_offset,
    const uint32_t* d_ff_values,
    const uint32_t  face_id,
    uint32_t&       len) {
    uint32_t start = 0;
    if (face_id != 0) {
        start = d_ff_offset[face_id - 1];
    }
    len = d_ff_offset[face_id] - start;

    return d_ff_values + start;
}

/**
 * @brief initialize every seed face's patch as tid
 */
__global__ static void write_initial_face_patch(const uint32_t num_seeds,
                                                uint32_t*      d_face_patch,
                                                uint32_t*      d_seeds,
                                                uint32_t*      d_patches_size) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < num_seeds) {
        uint32_t seed      = d_seeds[tid];
        d_face_patch[seed] = tid << 1;
        assert(d_patches_size[tid] == 0);
        d_patches_size[tid] = 1;
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief set new-added-to-queue vertices in the last propagation to the source of next propagation
 * 
 */
__global__ static void reset_queue_ptr(uint32_t* d_queue_ptr) {
    d_queue_ptr[0] = d_queue_ptr[1];
    d_queue_ptr[1] = d_queue_ptr[2];
}

/**
 * @brief propagation process for one-ring of vertex in queue
 * 
 */
__global__ static void cluster_seed_propagation(const uint32_t  num_faces,
                                                const uint32_t  num_patches,
                                                uint32_t*       d_queue_ptr,
                                                uint32_t*       d_queue,
                                                uint32_t*       d_face_patch,
                                                uint32_t*       d_patches_size,
                                                const uint32_t* d_ff_offset,
                                                const uint32_t* d_ff_values) {
    // the first bit in d_face_patch is reserved for 'is boundary face'
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t current_queue_end   = d_queue_ptr[1];
    uint32_t current_queue_start = d_queue_ptr[0];
    while (tid >= current_queue_start && tid < current_queue_end) {
        uint32_t        face_id    = d_queue[tid];
        uint32_t        face_patch = d_face_patch[face_id] >> 1;
        uint32_t        ff_len     = 0;
        const uint32_t* ff_ptr =
            get_face_faces(d_ff_offset, d_ff_values, face_id, ff_len);

        uint32_t is_boundary = 0;

        for (uint32_t i = 0; i < ff_len; i++) {
            uint32_t n_face = ff_ptr[i];

            // if the neighbor is not assign to any patch yet,
            // assign it to the same patch as this vertex.
            uint32_t assumed = ::atomicCAS(
                &d_face_patch[n_face], INVALID32, (face_patch << 1));
            assert((assumed >> 1) < num_patches || assumed == INVALID32);

            if (assumed == INVALID32) {
                // add to queue
                uint32_t old = ::atomicAdd(&d_queue_ptr[2], uint32_t(1));
                d_queue[old] = n_face;
                assert(old < num_faces);

                old = ::atomicAdd(&d_patches_size[face_patch], uint32_t(1));
                assert(old < num_faces);

            } else {
                // whether it is of boundry
                if ((assumed >> 1) != face_patch) {
                    is_boundary = is_boundary | 1;
                }
            }
        }

        face_patch            = face_patch << 1;
        face_patch            = face_patch | is_boundary;
        d_face_patch[face_id] = face_patch;

        tid += blockDim.x * gridDim.x;
    }
}

__global__ static void construct_patches_compressed(
    const uint32_t  num_faces,
    const uint32_t* d_face_patch,
    const uint32_t  num_patches,
    const uint32_t* d_patches_offset,
    uint32_t*       d_patches_size,
    uint32_t*       d_patches_val) {
    uint32_t face = threadIdx.x + blockIdx.x * blockDim.x;
    // use d_patches_offset to calculate d_patches_size and d_patches_val
    while (face < num_faces) {
        uint32_t patch_id    = d_face_patch[face];
        uint32_t is_boundary = patch_id & 1;
        patch_id             = patch_id >> 1;
        uint32_t pos = ::atomicAdd(&d_patches_size[patch_id], uint32_t(1));
        if (patch_id != 0) {
            pos += d_patches_offset[patch_id - 1];
        }
        uint32_t res = face << 1;
        res          = res | is_boundary;
        assert(pos < num_faces);
        assert(face < ((num_faces << 1) | 1));
        d_patches_val[pos] = res;
        face += blockDim.x * gridDim.x;
    }
}

/**
 * @brief decide the seed of each patch
 * 
 */
__global__ static void interior(const uint32_t  num_patches,
                                const uint32_t* d_patches_offset,
                                const uint32_t* d_patches_val,
                                const uint32_t* d_face_patch,
                                uint32_t*       d_seeds,
                                const uint32_t* d_ff_offset,
                                const uint32_t* d_ff_values,
                                uint32_t*       d_queue) {
    if (blockIdx.x < num_patches) {
        // one block per patch
        __shared__ uint32_t s_queue_size;
        if (threadIdx.x == 0) {
            s_queue_size = 0;
        }

        const uint32_t patch_id = blockIdx.x;
        const uint32_t p_start =
            (patch_id == 0) ? 0 : d_patches_offset[patch_id - 1];
        const uint32_t p_end  = d_patches_offset[patch_id];
        const uint32_t p_size = p_end - p_start;
        uint32_t       tid    = threadIdx.x;

        extern __shared__ uint32_t s_queue[];


        // construct boundary queue
        // one thread per face
        tid = threadIdx.x;
        while (tid < p_size) {
            uint32_t face = d_patches_val[tid + p_start];
            if (face & 1) {
                // is boundary
                uint32_t pos = ::atomicAdd(&s_queue_size, uint32_t(1));
                assert(s_queue_size <= p_size);
                face = face >> 1;
                s_queue[pos]  = face;
                d_queue[face] = 0;
            }
            tid += blockDim.x;
        }
        __syncthreads();

        // if there is no boundary, it means that the patch is a single
        // component. Pick any face as a seed.
        if (s_queue_size > 0) {
            uint32_t queue_end   = 0;
            uint32_t queue_start = 0;
            // from the outmost ring to the center
            while (true) {
                queue_start = queue_end;
                queue_end   = s_queue_size;
                __syncthreads();

                if (queue_end == p_size) {
                    break;
                }

                tid = threadIdx.x;
                while (tid < queue_end - queue_start) {
                    uint32_t        face   = s_queue[tid + queue_start];
                    uint32_t        ff_len = 0;
                    const uint32_t* ff_ptr =
                        get_face_faces(d_ff_offset, d_ff_values, face, ff_len);
                    for (uint32_t i = 0; i < ff_len; ++i) {
                        uint32_t n_face = ff_ptr[i];
                        if (d_face_patch[n_face] >> 1 == patch_id) {
                            uint32_t assumed = ::atomicCAS(
                                d_queue + n_face, INVALID32, patch_id);
                            if (assumed == INVALID32) {
                                uint32_t pos =
                                    ::atomicAdd(&s_queue_size, uint32_t(1));
                                assert(s_queue_size <= p_size);
                                s_queue[pos] = n_face;
                            }
                        }
                    }
                    tid += blockDim.x;
                }
                __syncthreads();
            }

            assert(queue_end == p_size);
            // pick random face
            if (threadIdx.x == 0) {
                // the last one is about at the center of the patch
                if (queue_start != 0) {
                    d_seeds[patch_id] = s_queue[queue_start];
                }
            }
        }
    }
}


__global__ static void add_more_seeds(const uint32_t  num_patches,
                                      uint32_t*       d_new_num_patches,
                                      uint32_t*       d_seeds,
                                      const uint32_t* d_patches_offset,
                                      const uint32_t* d_patches_val,
                                      const uint32_t  threshold) {
    if (blockIdx.x < num_patches) {
        uint32_t       patch_id = blockIdx.x;
        const uint32_t p_start =
            (patch_id == 0) ? 0 : d_patches_offset[patch_id - 1];
        const uint32_t p_end  = d_patches_offset[patch_id];
        const uint32_t p_size = p_end - p_start;

        if (p_size > threshold) {
            // need to add seed
            if (threadIdx.x == 0) {
                // look for a boundary face
                // add only one seed for a patch a time
                for (uint32_t f = p_start; f < p_end; ++f) {
                    uint32_t face = d_patches_val[f];
                    if (face & 1) {
                        // is boundary
                        uint32_t new_patch_id = ::atomicAdd(d_new_num_patches, 1u);
                        d_seeds[new_patch_id] = face >> 1;
                        break;
                    }
                }
            }
        }
    }
}
}  // namespace rxmesh