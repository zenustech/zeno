#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_discontinuity.cuh>
#include "rxmesh_queries.cuh"
#include "../context.h"
#include "../handle.h"
#include "../iterator.cuh"
#include "../utils/util.h"

namespace zeno::rxmesh {
namespace detail {
template <Op op, uint32_t blockThreads, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const PatchInfo& patch_info,
                                                  activeSetT compute_active_set,
                                                  const bool oriented,
                                                  uint32_t&  num_src_in_patch,
                                                  uint16_t*& s_output_offset,
                                                  uint16_t*& s_output_value,
                                                  uint16_t&  num_owned,
                                                  uint32_t*& not_owned_patch,
                                                  uint16_t*& not_owned_local_id) {
    static_assert(op != Op::EE, "Op::EE is not supported!");

    num_src_in_patch = 0;
    if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
        num_src_in_patch = patch_info.num_owned_vertices;
    }
    if constexpr (op == Op::EV || op == Op::EF) {
        num_src_in_patch = patch_info.num_owned_edges;
    }
    if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
        num_src_in_patch = patch_info.num_owned_faces;
    }

    bool     is_active = false;
    uint16_t local_id  = threadIdx.x;
    while (local_id < num_src_in_patch) {
        is_active = is_active || compute_active_set({patch_info.patch_id, local_id});
        local_id += blockThreads;
    }

    if (__syncthreads_or(is_active) == 0) {
        // this block/patch has no work to do
        num_src_in_patch = 0;
        return;
    }

    // 2) Load the patch info
    // TODO need shift shrd_mem to be aligned to 128-byte boundary
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_ev = shrd_mem;
    uint16_t*                  s_fe = shrd_mem;
    load_mesh_async<op>(patch_info, s_ev, s_fe, true);

    not_owned_patch    = reinterpret_cast<uint32_t*>(shrd_mem);
    not_owned_local_id = shrd_mem;
    num_owned          = 0;

    __syncthreads();

    // 3)Perform the query operation
    if (oriented) {
        assert(op == Op::VV);
        if constexpr (op == Op::VV) {        
            v_v_oriented<blockThreads>(
                patch_info, s_output_offset, s_output_value, s_ev);
        }
    } else {
        if constexpr (!(op == Op::VV || op == Op::FV || op == Op::FF)) {
            load_not_owned_async<op>(patch_info,
                                     not_owned_local_id,
                                     not_owned_patch,
                                     num_owned,
                                     true);
        }
        
        query<blockThreads, op>(s_output_offset,
                                s_output_value,
                                s_ev,
                                s_fe,
                                patch_info.num_vertices,
                                patch_info.num_edges,
                                patch_info.num_faces);
    }

    // load not-owned local and patch id
    if constexpr (op == Op::VV || op == Op::FV || op == Op::FF) {
        // need to sync since we will overwrite things that are used in query
        __syncthreads();
        load_not_owned_async<op>(
            patch_info, not_owned_local_id, not_owned_patch, num_owned, true);
    }

    __syncthreads();
}

template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  const uint32_t patch_id,
                                                  computeT       compute_op,
                                                  activeSetT compute_active_set,
                                                  const bool oriented = false) {
    // Extract the type of the input parameters of the compute lambda function.
    // The first parameter should be ElementHandle and second parameter
    // should be RXMesh ElementIterator

    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeHandleT   = typename ComputeTraits::template arg<0>::type;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;
    using LocalT           = typename ComputeIteratorT::LocalT;

    // Extract the type of the single input parameter of the active_set lambda
    // function. It should be ElementHandle and it should match the
    // first parameter of the compute lambda function
    using ActiveSetTraits  = detail::FunctionTraits<activeSetT>;
    using ActiveSetHandleT = typename ActiveSetTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<ActiveSetHandleT, ComputeHandleT>,
        "First argument of compute_op lambda function should match the first "
        "argument of active_set lambda function ");

    static_assert(op != Op::EE, "Op::EE is not supported!");


    assert(patch_id < context.get_num_patches());

    uint32_t  num_src_in_patch = 0;
    uint16_t* s_output_offset(nullptr);
    uint16_t* s_output_value(nullptr);
    uint16_t  num_owned;
    uint32_t* not_owned_patch(nullptr);
    uint16_t* not_owned_local_id(nullptr);

    query_block_dispatcher<op, blockThreads>(
        context.get_patches_info()[patch_id],
        compute_active_set,
        oriented,
        num_src_in_patch,
        s_output_offset,
        s_output_value,
        num_owned,
        not_owned_patch,
        not_owned_local_id);

    // Call compute on the output in shared memory by looping over all
    // source elements in this patch.

    uint16_t local_id = threadIdx.x;
    while (local_id < num_src_in_patch) {

        assert(s_output_value);

        if (compute_active_set({patch_id, local_id})) {
            constexpr uint32_t fixed_offset =
                ((op == Op::EV)                 ? 2 :
                 (op == Op::FV || op == Op::FE) ? 3 :
                                                  0);


            ComputeHandleT   handle(patch_id, local_id);
            ComputeIteratorT iter(local_id,
                                  reinterpret_cast<LocalT*>(s_output_value),
                                  s_output_offset,
                                  fixed_offset,
                                  patch_id,
                                  num_owned,
                                  not_owned_patch,
                                  not_owned_local_id,
                                  int(op == Op::FE));

            compute_op(handle, iter);
        }

        local_id += blockThreads;
    }
}

}  // namespace detail
/**
 * @brief The main query function to be called by the whole block. In this
 * function, threads will be assigned to mesh elements which will be accessible
 * through the input computation lambda function (compute_op). This function
 * also provides a predicate to specify the active set i.e., the set on which
 * the query operations should be done. This is mainly used to skip query on
 * a subset of the input mesh elements which may lead to better performance
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @tparam activeSetT the type of active set lambda function (inferred)
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param compute_op the computation lambda function that will be executed by
 * each thread in the block. This lambda function takes two input parameters:
 * 1. Handle to the mesh element assigned to the thread. The handle type matches
 * the source of the query 2. an iterator to the query output.
 * The iterator type matches the type of the mesh element
 * "iterated" on 
 * @param compute_active_set a predicate used to specify the active set. This
 * lambda function take a single parameter which is a handle of the type similar
 * to the input of the query operation
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT, typename activeSetT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  computeT       compute_op,
                                                  activeSetT compute_active_set,
                                                  const bool oriented = false) {
    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }

    detail::query_block_dispatcher<op, blockThreads>(
        context, blockIdx.x, compute_op, compute_active_set, oriented);
}


/**
 * @brief The main query function to be called by the whole block. In this
 * function, threads will be assigned to mesh elements which will be accessible
 * through the input computation lambda function (compute_op).
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param compute_op the computation lambda function that will be executed by
 * each thread in the block. This lambda function takes two input parameters:
 * 1. Handle to the mesh element assigned to the thread. The handle type matches
 * the source of the query 2. an iterator to
 * the query output. The iterator type matches the type of the mesh element
 * "iterated" on
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT>
__device__ __inline__ void query_block_dispatcher(const Context& context,
                                                  computeT       compute_op,
                                                  const bool oriented = false) {
    // Extract the type of the first input parameters of the compute lambda
    // function. It should be ElementHandle
    using ComputeTraits  = detail::FunctionTraits<computeT>;
    using ComputeHandleT = typename ComputeTraits::template arg<0>::type;

    query_block_dispatcher<op, blockThreads>(
        context, compute_op, [](ComputeHandleT) { return true; }, oriented);
}


/**
 * @brief This function is used to perform a query operation on a specific mesh
 * element. This is only needed for higher query (e.g., 2-ring query) where the
 * first query is done using query_block_dispatcher in which each thread is
 * assigned to a mesh element. Subsequent queries should be handled by this
 * function. This function should be called by the whole CUDA block.
 * @tparam Op the type of query operation
 * @tparam blockThreads the number of CUDA threads in the block
 * @tparam computeT the type of compute lambda function (inferred)
 * @tparam HandleT the type of input handle (inferred) which should match the
 * input of the query operations
 * @param context which store various parameters needed for the query
 * operation. The context can be obtained from RXMeshStatic
 * @param src_id the input mesh element to the query. Inactive threads can
 * simply pass HandleT() in which case they are skipped
 * @param compute_op the computation lambda function that will be executed by
 * the thread. This lambda function takes two input parameters:
 * 1. HandleT which is the same as src_id 2. an iterator to the query output.
 * The iterator type matches the type of the mesh element "iterated" on
 * @param oriented specifies if the query are oriented. Currently only VV query
 * is supported for oriented queries. FV, FE and EV is oriented by default
 */
template <Op op, uint32_t blockThreads, typename computeT, typename HandleT>
__device__ __inline__ void higher_query_block_dispatcher(
    const Context& context,
    const HandleT  src_id,
    computeT       compute_op,
    const bool     oriented = false) {
    using ComputeTraits    = detail::FunctionTraits<computeT>;
    using ComputeIteratorT = typename ComputeTraits::template arg<1>::type;

    // The whole block should be calling this function. If one thread is not
    // participating, its src_id should be INVALID32

    auto compute_active_set = [](HandleT) { return true; };

    // the source and local id of the source mesh element
    std::pair<uint32_t, uint16_t> pl = src_id.unpack();

    // Here, we want to identify the set of unique patches for this thread
    // block. We do this by first sorting the patches, compute discontinuity
    // head flag, then threads with head flag =1 can add their patches to the
    // shared memory buffer that will contain the unique patches

    __shared__ uint32_t s_block_patches[blockThreads];
    __shared__ uint32_t s_num_patches;
    if (threadIdx.x == 0) {
        s_num_patches = 0;
    }
    typedef cub::BlockRadixSort<uint32_t, blockThreads, 1>  BlockRadixSort;
    typedef cub::BlockDiscontinuity<uint32_t, blockThreads> BlockDiscontinuity;
    union TempStorage
    {
        typename BlockRadixSort::TempStorage     sort_storage;
        typename BlockDiscontinuity::TempStorage discont_storage;
    };
    __shared__ TempStorage all_temp_storage;
    uint32_t               thread_data[1], thread_head_flags[1];
    thread_data[0]       = pl.first;
    thread_head_flags[0] = 0;
    BlockRadixSort(all_temp_storage.sort_storage).Sort(thread_data);
    BlockDiscontinuity(all_temp_storage.discont_storage)
        .FlagHeads(thread_head_flags, thread_data, cub::Inequality());

    if (thread_head_flags[0] == 1 && thread_data[0] != INVALID32) {
        uint32_t id         = atomicAdd(&s_num_patches, uint32_t(1));
        s_block_patches[id] = thread_data[0];
    }

    // We could eliminate the discontinuity operation and atomicAdd and instead
    // use thrust::unique. However, this method causes illegal memory access
    // and it looks like a bug in thrust
    __syncthreads();


    for (uint32_t p = 0; p < s_num_patches; ++p) {

        uint32_t patch_id = s_block_patches[p];

        assert(patch_id < context.get_num_patches());

        uint32_t  num_src_in_patch = 0;
        uint16_t *s_output_offset(nullptr), *s_output_value(nullptr);
        uint16_t  num_owned = 0;
        uint16_t* not_owned_local_id(nullptr);
        uint32_t* not_owned_patch(nullptr);

        detail::template query_block_dispatcher<op, blockThreads>(
            context.get_patches_info()[patch_id],
            compute_active_set,
            oriented,
            num_src_in_patch,
            s_output_offset,
            s_output_value,
            num_owned,
            not_owned_patch,
            not_owned_local_id);


        if (pl.first == patch_id) {

            constexpr uint32_t fixed_offset =
                ((op == Op::EV)                 ? 2 :
                 (op == Op::FV || op == Op::FE) ? 3 :
                                                  0);

            ComputeIteratorT iter(
                pl.second,
                reinterpret_cast<typename ComputeIteratorT::LocalT*>(
                    s_output_value),
                s_output_offset,
                fixed_offset,
                patch_id,
                num_owned,
                not_owned_patch,
                not_owned_local_id,
                int(op == Op::FE));

            compute_op(src_id, iter);
        }
        __syncthreads();
    }
}


}  // namespace rxmesh
