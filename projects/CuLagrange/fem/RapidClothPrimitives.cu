#include "RapidCloth.cuh"

namespace zeno{

template<class ValT, class tvT>
ValT RapidClothSystem::tvMax(zs::CudaExecutionPolicy &cudaPol, const tvT& tv, const zs::SmallString& tag, 
    std::size_t n, zs::wrapt<ValT> valWrapT) {
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    auto nwarps = count_warps(n); 
    constexpr auto is_val_int = zs::is_same_v<ValT, int>; 
    zs::conditional_t<is_val_int, decltype(itemp), decltype(temp)>& temp = is_val_int ? itemp: temp; 
    temp.resize(nwarps); 
    cudaPol(range(nwarps * 32), 
        [data = view<space>({}, tv), res = view<space>(temp), 
        n, tagOff = tv.getPropertyOffset(tag), valWrapT] __device__ (int pi) mutable {
            auto val = limits<ValT>::min(); 
            if (pi < n)
                val = data(tagOff, pi, valWrapT); 
#if __CUDA_ARCH__ >= 800
            auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block()); 
            auto ret = zs::cg::reduce(tile, val, zs::cg::greater<ValT>()); 
            if (tile.thread_rank() == 0)
                res[pi / 32] = ret; 
#else 
            auto [mask, numValid] = warp_mask(pi, n); 
            auto locid = threadIdx.x & 31; 
            for (int stride = 1; stride < 32; stride <<= 1) {
                auto tmp = __shfl_down_sync(mask, val, stride); 
                if (locid + stride < numValid)
                    val = val > tmp ? val : tmp; 
            }
            if (locid == 0)
                res[pi / 32] = val; 
#endif 
        }); 
    return reduce(cudaPol, temp, thrust::maximum<ValT>{}); 
}

template int RapidClothSystem::tvMax<int, typename RapidClothSystem::itiles_t>(
    zs::CudaExecutionPolicy&, const typename RapidClothSystem::itiles_t& tv, 
    const zs::SmallString&, std::size_t, zs::wrapt<int>); 

template<class ValT, class tvT>
ValT RapidClothSystem::tvMin(zs::CudaExecutionPolicy &cudaPol, const tvT& tv, const zs::SmallString& tag, 
    std::size_t n, zs::wrapt<ValT> valWrapT) {
    using namespace zs; 
    constexpr auto space = execspace_e::cuda; 
    auto nwarps = count_warps(n); 
    // constexpr auto is_val_int = zs::is_same_v<ValT, int>; 
    // auto &temp = this->temp; 
    // zs::conditional_t<is_val_int, decltype(itemp), decltype(this->temp)>& temp = is_val_int ? itemp: this->temp; 
    temp.resize(nwarps); 
    cudaPol(range(nwarps * 32), 
        [data = view<space>({}, tv), res = view<space>(temp), 
        n, tagOff = tv.getPropertyOffset(tag), valWrapT] __device__ (int pi) mutable {
            auto val = limits<ValT>::max(); 
            if (pi < n)
                val = data(tagOff, pi, valWrapT); 
#if __CUDA_ARCH__ >= 800
            auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block()); 
            auto ret = zs::cg::reduce(tile, val, zs::cg::less<ValT>()); 
            if (tile.thread_rank() == 0)
            {
                // printf("tvMin local result: %f at tile_idx = %d\n", 
                //     (float)ret, pi / 32); 
                res[pi / 32] = ret; 
            }
#else 
            auto [mask, numValid] = warp_mask(pi, n); 
            auto locid = threadIdx.x & 31; 
            for (int stride = 1; stride < 32; stride <<= 1) {
                auto tmp = __shfl_down_sync(mask, val, stride); 
                if (locid + stride < numValid)
                    val = val < tmp ? val : tmp; 
            }
            if (locid == 0)
                res[pi / 32] = val; 
#endif 
        }); 
    return reduce(cudaPol, temp, limits<ValT>::max(), thrust::minimum<ValT>{}); 
}

template typename RapidClothSystem::T RapidClothSystem::tvMin<typename RapidClothSystem::T, typename RapidClothSystem::tiles_t>(
    zs::CudaExecutionPolicy&, const typename RapidClothSystem::tiles_t& tv, 
    const zs::SmallString&, std::size_t, zs::wrapt<typename RapidClothSystem::T>); 


template <int codim>
typename RapidClothSystem::T RapidClothSystem::infNorm(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString& tag, 
    std::size_t n, zs::wrapv<codim>) {
    using namespace zs;
    using T = typename RapidClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(n);
    temp.resize(nwarps);
    cudaPol(range(nwarps * 32), [data = view<space>({}, vtemp), res = view<space>(temp), n = n,
                             offset = vtemp.getPropertyOffset(tag)] __device__(int pi) mutable {
        T val = 0; 
        if (pi < n)
        {
            auto v = data.pack(dim_c<codim>, offset, pi);
            val = v.abs().max();            
        }

#if __CUDA_ARCH__ >= 800
        auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
        auto ret = zs::cg::reduce(tile, val, zs::cg::greater<T>());
        if (tile.thread_rank() == 0)
            res[pi / 32] = ret;
#else
        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
#endif
    });
    return reduce(cudaPol, temp, thrust::maximum<T>{});
}
template typename RapidClothSystem::T RapidClothSystem::infNorm<3>(
    zs::CudaExecutionPolicy&, const zs::SmallString&, std::size_t, zs::wrapv<3>); 
template typename RapidClothSystem::T RapidClothSystem::infNorm<1>(
    zs::CudaExecutionPolicy&, const zs::SmallString&, std::size_t, zs::wrapv<1>); 

typename RapidClothSystem::T RapidClothSystem::l2Norm(zs::CudaExecutionPolicy &pol, const zs::SmallString &tag, std::size_t n) {
    return zs::sqrt(dot(pol, tag, tag, n));
}

typename RapidClothSystem::T RapidClothSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString &tag0,
                                                 const zs::SmallString &tag1, std::size_t n) {
    using namespace zs;
    using T = typename RapidClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(n);
    temp.resize(nwarps);
    temp.reset(0);
    cudaPol(range(n), [data = view<space>({}, vtemp), res = view<space>(temp), n = n,
                             offset0 = vtemp.getPropertyOffset(tag0),
                             offset1 = vtemp.getPropertyOffset(tag1)] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, offset0, pi);
        auto v1 = data.pack(dim_c<3>, offset1, pi);
        reduce_to(pi, n, v0.dot(v1), res[pi / 32]);
    });
    return reduce(cudaPol, temp, thrust::plus<T>{});
}
}