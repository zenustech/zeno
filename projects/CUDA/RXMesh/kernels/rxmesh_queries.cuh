#pragma once

#include <assert.h>
#include <cub/block/block_scan.cuh>
#include "loader.cuh"
#include "../utils/util.cuh"

namespace zeno::rxmesh {
namespace detail {

/**
 * @brief Compute block-wide exclusive sum using CUB
 */
template <typename T, uint32_t blockThreads>
__device__ __forceinline__ void cub_block_exclusive_sum(T*             data,
                                                        const uint32_t size) {
    __shared__ T s_prv_run_aggregate;
    if (threadIdx.x == 0) {
        s_prv_run_aggregate = 0;
    }

    const uint32_t num_run = (size + blockThreads - 1) / blockThreads;
    typedef cub::BlockScan<T, blockThreads>    BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    for (uint32_t r = 0; r < num_run; ++r) {
        T        value = 0;
        T        run_aggregate;
        uint32_t index = r * blockThreads + threadIdx.x;

        if (index < size) {
            value = data[index];
        }

        BlockScan(temp_storage).ExclusiveSum(value, value, run_aggregate);

        if (index < size) {
            if (r > 0) {
                value += s_prv_run_aggregate;
            }
            data[index] = value;
        }

        __syncthreads();
        if (threadIdx.x == blockThreads - 1) {
            if (r == num_run - 1) {
                data[size] = s_prv_run_aggregate + run_aggregate;
            } else {
                s_prv_run_aggregate += run_aggregate;
            }
        }
    }
}

template <uint32_t rowOffset, uint32_t blockThreads>
__device__ __forceinline__ void block_mat_transpose(const uint32_t num_rows,
                                                    const uint32_t num_cols,
                                                    uint16_t*      mat,
                                                    uint16_t*      output,
                                                    int            shift = 0) {
    // 1) Load mat into registers and zero out mat
    uint16_t thread_data[TRANSPOSE_ITEM_PER_THREAD];
    uint16_t local_offset[TRANSPOSE_ITEM_PER_THREAD];
    uint32_t nnz = num_rows * rowOffset;

    for (uint32_t i = 0; i < TRANSPOSE_ITEM_PER_THREAD; ++i) {
        uint32_t index = TRANSPOSE_ITEM_PER_THREAD * threadIdx.x + i;
        if (index < nnz) {
            thread_data[i] = mat[index] >> shift;
        } else {
            thread_data[i] = INVALID16;
        }
    }

    uint32_t m = nnz > num_cols ? nnz : num_cols;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < m; i += blockThreads) {
        mat[i] = 0;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 700
    // 2) compute the number of items in each bucket/col
    __half* mat_half = (__half*)(mat);
    for (uint32_t i = 0; i < TRANSPOSE_ITEM_PER_THREAD; ++i) {
        if (thread_data[i] != INVALID16) {
            local_offset[i] = atomicAdd(&mat_half[thread_data[i]], 1);
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < num_cols; i += blockThreads) {
        uint16_t val = uint16_t(mat_half[i]);
        mat[i]       = val;
    }
#else
    for (uint32_t i = 0; i < TRANSPOSE_ITEM_PER_THREAD; ++i) {
        if (thread_data[i] != INVALID16) {
            local_offset[i] = atomicAdd(&mat[thread_data[i]], 1u);
        } else {
            break;
        }
    }
    __syncthreads();
#endif


    // 3) exclusive scan on mat to compute the offset
    cub_block_exclusive_sum<uint16_t, blockThreads>(mat, num_cols);

    // 4) actually write the values
    for (uint32_t i = 0; i < TRANSPOSE_ITEM_PER_THREAD; ++i) {
        uint16_t item = thread_data[i];
        if (item != INVALID16) {
            uint16_t offset = mat[item] + local_offset[i];
            uint16_t row    = (TRANSPOSE_ITEM_PER_THREAD * threadIdx.x + i) / rowOffset;
            output[offset]  = row;
        } else {
            break;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void e_f_manifold(const uint16_t  num_edges,
                                             const uint16_t  num_faces,
                                             const uint16_t* s_fe,
                                             uint16_t*       s_ef) {
    // s_ef should be filled with INVALID16 before calling this function

    for (uint16_t e = threadIdx.x; e < 3 * num_faces; e += blockThreads) {
        uint16_t edge    = s_fe[e] >> 1;
        uint16_t face_id = e / 3;

        auto ret = atomicCAS(s_ef + 2 * edge, INVALID16, face_id);
        if (ret != INVALID16) {
            ret = atomicCAS(s_ef + 2 * edge + 1, INVALID16, face_id);
            assert(ret == INVALID16);
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_v_oriented(const PatchInfo& patch_info,
                                             uint16_t*&       s_output_offset,
                                             uint16_t*&       s_output_value,
                                             uint16_t*        s_ev) {

    const uint16_t num_edges          = patch_info.num_edges;
    const uint16_t num_faces          = patch_info.num_faces;
    const uint16_t num_vertices       = patch_info.num_vertices;
    const uint16_t num_owned_vertices = patch_info.num_owned_vertices;

    s_output_offset = &s_ev[0];
    s_output_value  = &s_ev[num_vertices + 1 + (num_vertices + 1) % 2];

    // start by loading the faces while also doing transposing EV (might
    // increase ILP)
    uint16_t*   s_fe    = &s_output_value[2 * num_edges];
    uint16_t*   s_ef    = &s_fe[3 * num_faces + (3 * num_faces) % 2];
    LocalIndexT* temp_fe = reinterpret_cast<LocalIndexT*>(s_fe);
    load_async(reinterpret_cast<const uint16_t*>(patch_info.fe),
               num_faces * 3,
               reinterpret_cast<uint16_t*>(temp_fe),
               true);

    for (uint32_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        s_ef[i] = INVALID16;
    }

    block_mat_transpose<2u, blockThreads>(
        num_edges, num_vertices, s_output_offset, s_output_value);

    // block_mat_transpose<2u, blockThreads>(
    //    num_faces, num_edges, s_patch_EF_offset, s_patch_EF_output);

    // We could have used block_mat_transpose to transpose FE so we can look
    // up the "two" faces sharing an edge. But we can do better because we know
    // that we are working on manifold so it is only two edges per face. We
    // also wanna keep FE for quick look up on a face's three edges.

    // We need to sync here to make sure that s_fe is loaded but there is
    // a sync in block_mat_transpose that takes care of this

    e_f_manifold<blockThreads>(num_edges, num_faces, s_fe, s_ef);

    // To orient, we pin the first edge and check all the subsequent edges
    // For each edge, we search for the two faces containing it (should be
    // only two faces since this is a manifold mesh).
    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_owned_vertices; v += blockDim.x) {

        // if the vertex is not owned by this patch, then there is no reason
        // to orient its edges because no serious computation is done on it

        uint16_t start = s_output_offset[v];
        uint16_t end   = s_output_offset[v + 1];


        for (uint16_t e_id = start; e_id < end - 1; ++e_id) {
            uint16_t e_0 = s_output_value[e_id];
            uint16_t f0(s_ef[2 * e_0]), f1(s_ef[2 * e_0 + 1]);

            // we don't do it for boundary faces
            assert(f0 != INVALID16 && f1 != INVALID16 && f0 < num_faces &&
                   f1 < num_faces);


            // candidate next edge (only one of them will win)
            uint16_t e_candid_0, e_candid_1;

            if ((s_fe[3 * f0 + 0] >> 1) == e_0) {
                e_candid_0 = s_fe[3 * f0 + 2] >> 1;
            }
            if ((s_fe[3 * f0 + 1] >> 1) == e_0) {
                e_candid_0 = s_fe[3 * f0 + 0] >> 1;
            }
            if ((s_fe[3 * f0 + 2] >> 1) == e_0) {
                e_candid_0 = s_fe[3 * f0 + 1] >> 1;
            }

            if ((s_fe[3 * f1 + 0] >> 1) == e_0) {
                e_candid_1 = s_fe[3 * f1 + 2] >> 1;
            }
            if ((s_fe[3 * f1 + 1] >> 1) == e_0) {
                e_candid_1 = s_fe[3 * f1 + 0] >> 1;
            }
            if ((s_fe[3 * f1 + 2] >> 1) == e_0) {
                e_candid_1 = s_fe[3 * f1 + 1] >> 1;
            }

            for (uint16_t vn = e_id + 1; vn < end; ++vn) {
                uint16_t e_winning_candid = s_output_value[vn];
                if (e_candid_0 == e_winning_candid ||
                    e_candid_1 == e_winning_candid) {
                    uint16_t temp            = s_output_value[e_id + 1];
                    s_output_value[e_id + 1] = e_winning_candid;
                    s_output_value[vn]       = temp;
                    break;
                }
            }
        }
    }

    __syncthreads();

    // Load EV into s_ef since both has the same size (2*#E)
    s_ev                  = s_ef;
    LocalIndexT* temp_ev = reinterpret_cast<LocalIndexT*>(s_ef);
    load_async(reinterpret_cast<const uint16_t*>(patch_info.ev),
               num_edges * 2,
               reinterpret_cast<uint16_t*>(temp_ev),
               true);

    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        uint32_t start = s_output_offset[v];
        uint32_t end   = s_output_offset[v + 1];


        for (uint32_t e = start; e < end; ++e) {
            uint16_t edge = s_output_value[e];
            uint16_t v0   = s_ev[2 * edge];
            uint16_t v1   = s_ev[2 * edge + 1];

            assert(v0 == v || v1 == v);
            s_output_value[e] = (v0 == v) * v1 + (v1 == v) * v0;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_e(const uint16_t num_vertices,
                                    const uint16_t num_edges,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_output) {
    // M_ve = M_ev^{T}.
    block_mat_transpose<2u, blockThreads>(
        num_edges, num_vertices, d_edges, d_output);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_v(const uint16_t num_vertices,
                                    const uint16_t num_edges,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_output) {
    // M_vv = M_EV^{T} \dot M_EV

    uint16_t* s_edges_duplicate = &d_edges[2 * 2 * num_edges];

    assert(2 * 2 * num_edges >= num_vertices + 1 + 2 * num_edges);

    for (uint16_t i = threadIdx.x; i < 2 * num_edges; i += blockThreads) {
        s_edges_duplicate[i] = d_edges[i];
    }

    // TODO we might be able to remove this sync if transpose has a sync
    // that is done before writing to mat
    __syncthreads();

    v_e<blockThreads>(num_vertices, num_edges, d_edges, d_output);

    __syncthreads();

    for (uint32_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        uint32_t start = d_edges[v];
        uint32_t end   = d_edges[v + 1];

        for (uint32_t e = start; e < end; ++e) {
            uint16_t edge = d_output[e];
            uint16_t v0   = s_edges_duplicate[2 * edge];
            uint16_t v1   = s_edges_duplicate[2 * edge + 1];

            assert(v0 == v || v1 == v);
            d_output[e] = (v0 == v) * v1 + (v1 == v) * v0;
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void f_v(const uint16_t  num_edges,
                                    const uint16_t* d_edges,
                                    const uint16_t  num_faces,
                                    uint16_t*       d_faces) {
    // M_FV = M_FE \dot M_EV

    for (uint32_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t f_v[3];
        uint32_t f_id = 3 * f;
        for (uint32_t i = 0; i < 3; i++) {
            uint16_t e = d_faces[f_id + i];
            flag_t   e_dir(0);
            Context::unpack_edge_dir(e, e, e_dir);
            // if the direction is flipped, we take the second vertex
            uint16_t e_id = (2 * e) + (1 * e_dir);
            assert(e_id < 2 * num_edges);
            f_v[i] = d_edges[e_id];
        }
        for (uint32_t i = 0; i < 3; i++) {
            d_faces[f * 3 + i] = f_v[i];
        }
    }
}

template <uint32_t blockThreads>
__device__ __forceinline__ void v_f(const uint16_t num_faces,
                                    const uint16_t num_edges,
                                    const uint16_t num_vertices,
                                    uint16_t*      d_edges,
                                    uint16_t*      d_faces) {
    // M_vf = M_ev^{T} \dot M_fe^{T} = (M_ev \dot M_fe)^{T} = M_fv^{T}

    // We follow the math here by computing M_fv and then transpose it
    // In doing so we reuse all the shared memory used to store d_edges
    // and d_faces
    // First M_fv is computing in place i.e., d_face will contain the
    // face vertices of each face (instead of edges)
    // Second, the transpose happens in place i.e., d_faces will hold the
    // offset and d_edges will hold the value (row id)

    f_v<blockThreads>(num_edges, d_edges, num_faces, d_faces);
    __syncthreads();

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_vertices, d_faces, d_edges);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void e_f(const uint16_t num_edges,
                                    const uint16_t num_faces,
                                    uint16_t*      d_faces,
                                    uint16_t*      d_output,
                                    int            shift = 1) {
    // M_ef = M_fe^{T}.
    // d_output contains the row id of the transpose matrix while d_faces
    // will contain the offset that starts withzero and end with num_faces*3.
    // d_output should be allocated to size = num_faces*3

    block_mat_transpose<3u, blockThreads>(
        num_faces, num_edges, d_faces, d_output, shift);
}

template <uint32_t blockThreads>
__device__ __forceinline__ void f_f(const uint16_t num_edges,
                                    const uint16_t num_faces,
                                    uint16_t*      s_FE,
                                    uint16_t*      s_FF_offset,
                                    uint16_t*      s_FF_output) {
    // First construct M_EF in shared memory

    uint16_t* s_EF_offset = &s_FE[num_faces * 3];
    uint16_t* s_EF_output = &s_EF_offset[num_edges + 1];

    // copy FE in to EF_offset so we can do the transpose in place without
    // losing FE
    for (uint16_t i = threadIdx.x; i < num_faces * 3; i += blockThreads) {
        uint16_t e     = s_FE[i] >> 1;
        s_EF_offset[i] = e;
        s_FE[i]        = e;
    }
    __syncthreads();

    e_f<blockThreads>(num_edges, num_faces, s_EF_offset, s_EF_output, 0);
    __syncthreads();

    // Every thread (T) is responsible for a face (F)
    // Each thread reads the edges (E) incident to its face (F). For each edge
    // (E), we read the "number" of incident faces (FF) to this edge (num_EF).
    // The number neighbor edges to the face F due to edge E is num_EF -1

    // TODO we can store this sum of neighbor faces in registers and then do
    // the exclusive sum on it and finally store it in shared memory
    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t num_neighbour_faces = 0;
        for (uint16_t e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];

            assert(s_EF_offset[edge + 1] >= s_EF_offset[edge]);

            num_neighbour_faces +=
                s_EF_offset[edge + 1] - s_EF_offset[edge] - 1;
        }
        s_FF_offset[f] = num_neighbour_faces;
    }
    __syncthreads();

    cub_block_exclusive_sum<uint16_t, blockThreads>(s_FF_offset, num_faces);

    for (uint16_t f = threadIdx.x; f < num_faces; f += blockThreads) {
        uint16_t offset = s_FF_offset[f];
        for (uint16_t e = 0; e < 3; ++e) {
            uint16_t edge = s_FE[3 * f + e];
            for (uint16_t ef = s_EF_offset[edge]; ef < s_EF_offset[edge + 1];
                 ++ef) {
                uint16_t n_face = s_EF_output[ef];
                if (n_face != f) {
                    s_FF_output[offset] = n_face;
                    ++offset;
                }
            }
        }
        assert(offset == s_FF_offset[f + 1]);
    }
}

template <uint32_t blockThreads, Op op>
__device__ __forceinline__ void query(uint16_t*&     s_output_offset,
                                      uint16_t*&     s_output_value,
                                      uint16_t*      s_ev,
                                      uint16_t*      s_fe,
                                      const uint16_t num_vertices,
                                      const uint16_t num_edges,
                                      const uint16_t num_faces) {
    switch (op) {
        case Op::VV: {
            assert(num_vertices <= 2 * num_edges);
            s_output_offset = &s_ev[0];
            s_output_value  = &s_ev[num_vertices + 1];
            v_v<blockThreads>(num_vertices, num_edges, s_ev, s_output_value);
            break;
        }
        case Op::VE: {
            assert(num_vertices <= 2 * num_edges);
            s_output_offset = &s_ev[0];
            s_output_value  = &s_ev[num_vertices + 1];
            v_e<blockThreads>(num_vertices, num_edges, s_ev, s_output_value);
            break;
        }
        case Op::VF: {
            assert(num_vertices <= 2 * num_edges);
            s_output_offset = &s_fe[0];
            s_output_value  = &s_ev[0];
            v_f<blockThreads>(num_faces, num_edges, num_vertices, s_ev, s_fe);
            break;
        }
        case Op::EV: {
            s_output_value = s_ev;
            break;
        }
        case Op::EF: {
            assert(num_edges <= 3 * num_faces);
            s_output_offset = &s_fe[0];
            s_output_value  = &s_fe[num_edges + 1];
            e_f<blockThreads>(num_edges, num_faces, s_fe, s_output_value);
            break;
        }
        case Op::FV: {
            s_output_value = s_fe;
            f_v<blockThreads>(num_edges, s_ev, num_faces, s_fe);
            break;
        }
        case Op::FE: {
            s_output_value = s_fe;
            break;
        }
        case Op::FF: {
            assert(num_edges <= 3 * num_faces);
            s_output_offset = &s_fe[3 * num_faces + 2 * 3 * num_faces];
            s_output_value = &s_output_offset[num_faces + 1];
            f_f<blockThreads>(
                num_edges, num_faces, s_fe, s_output_offset, s_output_value);

            break;
        }
        default:
            assert(1 != 1);
            break;
    }
}
}  // namespace detail
}  // namespace rxmesh
