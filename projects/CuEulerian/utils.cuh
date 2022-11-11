#pragma once

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

template <typename T> __forceinline__ __device__ void reduce_add(int i, int n, T val, T &dst) {
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
}

template <typename T> __forceinline__ __device__ void reduce_max(int i, int n, T val, T &dst) {
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

template <typename SpgPtrT>
inline auto src_tag(SpgPtrT spg, zs::SmallString attr_) {
    std::string attr = attr_.asString();
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}

template <typename SpgPtrT>
inline auto dst_tag(SpgPtrT spg, zs::SmallString attr_) {
    std::string atrr = attr_.asString();
    std::string metaTag = attr + "_cur";
    if (spg->hasMeta(metaTag)) {
        int cur = spg->readMeta<int>(metaTag);
        cur ^= 1;
        attr += std::to_string(cur);
    }
    return zs::SmallString{attr};
}

template <typename SpgPtrT>
inline void update_cur(SpgPtrT spg, zs::SmallString attr_) {
    std::string attr = attr_.asString();
    int &cur = spg->readMeta<int &>(attr + "_cur");
    cur ^= 1;
}

} // namespace zeno