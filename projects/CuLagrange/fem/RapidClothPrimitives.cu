#include "RapidCloth.cuh"

namespace zeno{

typename RapidClothSystem::T RapidClothSystem::infNorm(zs::CudaExecutionPolicy &cudaPol) {
    using namespace zs;
    using T = typename RapidClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(numDofs);
    temp.resize(nwarps);
    cudaPol(range(numDofs), [data = view<space>({}, vtemp), res = view<space>(temp), n = numDofs,
                             offset = vtemp.getPropertyOffset("dir")] __device__(int pi) mutable {
        auto v = data.pack(dim_c<3>, offset, pi);
        auto val = v.abs().max();

        auto [mask, numValid] = warp_mask(pi, n);
        auto locid = threadIdx.x & 31;
        for (int stride = 1; stride < 32; stride <<= 1) {
            auto tmp = __shfl_down_sync(mask, val, stride);
            if (locid + stride < numValid)
                val = zs::max(val, tmp);
        }
        if (locid == 0)
            res[pi / 32] = val;
    });
    return reduce(cudaPol, temp, thrust::maximum<T>{});
}

typename RapidClothSystem::T RapidClothSystem::l2Norm(zs::CudaExecutionPolicy &pol, const zs::SmallString tag) {
    return zs::sqrt(dot(pol, tag, tag));
}

typename RapidClothSystem::T RapidClothSystem::dot(zs::CudaExecutionPolicy &cudaPol, const zs::SmallString tag0,
                                                 const zs::SmallString tag1) {
    using namespace zs;
    using T = typename RapidClothSystem::T;
    constexpr auto space = execspace_e::cuda;
    auto nwarps = count_warps(numDofs);
    temp.resize(nwarps);
    temp.reset(0);
    cudaPol(range(numDofs), [data = view<space>({}, vtemp), res = view<space>(temp), n = numDofs,
                             offset0 = vtemp.getPropertyOffset(tag0),
                             offset1 = vtemp.getPropertyOffset(tag1)] __device__(int pi) mutable {
        auto v0 = data.pack(dim_c<3>, offset0, pi);
        auto v1 = data.pack(dim_c<3>, offset1, pi);
        reduce_to(pi, n, v0.dot(v1), res[pi / 32]);
    });
    return reduce(cudaPol, temp, thrust::plus<T>{});
}
}