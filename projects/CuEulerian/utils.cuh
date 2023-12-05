#pragma once
#if __CUDA_ARCH__ >= 800
#include <cooperative_groups/reduce.h>
#endif
#include "EulerianStructures.hpp"

namespace zeno {

/// utilities
constexpr std::size_t count_warps(std::size_t n) noexcept {
    return (n + 31) / 32;
}
constexpr int warp_index(int n) noexcept {
    return n / 32;
}
constexpr auto warp_mask(int i, int n) noexcept {
    int k = n % 32;
    const int tail = n - k;
    if (i < tail)
        return zs::make_tuple(0xFFFFFFFFu, 32);
    return zs::make_tuple(((unsigned)(1ull << k) - 1), k);
}

template <typename T>
__forceinline__ __device__ void reduce_add(int i, int n, T val, T &dst) {
#if __CUDA_ARCH__ >= 800
    auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
    auto ret = zs::cg::reduce(tile, val, zs::cg::plus<T>());
    if (tile.thread_rank() == 0)
        zs::atomic_add(zs::exec_cuda, &dst, ret);
#else
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val += tmp;
    }
    if (locid == 0)
        zs::atomic_add(zs::exec_cuda, &dst, val);
#endif
}

template <typename T>
__forceinline__ __device__ void reduce_max(int i, int n, T val, T &dst) {
#if __CUDA_ARCH__ >= 800
    auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
    auto ret = zs::cg::reduce(tile, val, zs::cg::greater<T>());
    if (tile.thread_rank() == 0)
        zs::atomic_max(zs::exec_cuda, &dst, ret);
#else
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val = zs::max(val, tmp);
    }
    if (locid == 0)
        zs::atomic_max(zs::exec_cuda, &dst, val);
#endif
}

template <typename T>
__forceinline__ __device__ void reduce_min(int i, int n, T val, T &dst) {
#if __CUDA_ARCH__ >= 800
    auto tile = zs::cg::tiled_partition<32>(zs::cg::this_thread_block());
    auto ret = zs::cg::reduce(tile, val, zs::cg::less<T>());
    if (tile.thread_rank() == 0)
        zs::atomic_min(zs::exec_cuda, &dst, ret);
#else
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val = zs::min(val, tmp);
    }
    if (locid == 0)
        zs::atomic_min(zs::exec_cuda, &dst, val);
#endif
}

template <typename T, typename Op = std::plus<T>>
inline T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res, Op op = {}) {
    using namespace zs;
    Vector<T> ret{res.get_allocator(), 1};
    bool shouldSync = cudaPol.shouldSync();
    cudaPol.sync(true);
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0, op);
    cudaPol.sync(shouldSync);
    return ret.getVal();
}

// sparse grid
inline auto src_tag(ZenoSparseGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}
inline auto src_tag(std::shared_ptr<ZenoSparseGrid> spg, zs::SmallString attr_) {
    return src_tag(spg.get(), attr_);
}

inline auto dst_tag(ZenoSparseGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        cur ^= 1;
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}
inline auto dst_tag(std::shared_ptr<ZenoSparseGrid> spg, zs::SmallString attr_) {
    return dst_tag(spg.get(), attr_);
}

inline void update_cur(ZenoSparseGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int &cur = spg->readMeta<int &>(metaTag);
        cur ^= 1;
    }
}
inline void update_cur(std::shared_ptr<ZenoSparseGrid> spg, zs::SmallString attr_) {
    update_cur(spg.get(), attr_);
}

// adaptive grid
inline auto src_tag(ZenoAdaptiveGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}
inline auto src_tag(std::shared_ptr<ZenoAdaptiveGrid> spg, zs::SmallString attr_) {
    return src_tag(spg.get(), attr_);
}

inline auto dst_tag(ZenoAdaptiveGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        cur ^= 1;
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}
inline auto dst_tag(std::shared_ptr<ZenoAdaptiveGrid> spg, zs::SmallString attr_) {
    return dst_tag(spg.get(), attr_);
}

inline void update_cur(ZenoAdaptiveGrid *spg, zs::SmallString attr_) {
    std::string attr = std::string(attr_);
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int &cur = spg->readMeta<int &>(metaTag);
        cur ^= 1;
    }
}
inline void update_cur(std::shared_ptr<ZenoAdaptiveGrid> spg, zs::SmallString attr_) {
    update_cur(spg.get(), attr_);
}

} // namespace zeno