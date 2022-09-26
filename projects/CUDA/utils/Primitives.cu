#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/parallel_reduce.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>

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

template <typename T, typename Op> __forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst, Op op) {
    auto [mask, numValid] = warp_mask(i, n);
    __syncwarp(mask);
    auto locid = threadIdx.x & 31;
    for (int stride = 1; stride < 32; stride <<= 1) {
        auto tmp = __shfl_down_sync(mask, val, stride);
        if (locid + stride < numValid)
            val = op(val, tmp);
    }
    if (locid == 0)
        dst = val;
}

template <typename TransOp, typename ReduceOp>
static float prim_reduce(typename ZenoParticles::particles_t &verts, float e, TransOp top, ReduceOp rop,
                         std::string attrToReduce) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    using T = typename ZenoParticles::particles_t::value_type;
    auto nchn = verts.getPropertySize(attrToReduce);
    auto offset = verts.getPropertyOffset(attrToReduce);
    const auto nwarps = count_warps(verts.size());

    auto cudaPol = cuda_exec().device(0);

    Vector<float> res{verts.get_allocator(), nwarps};
    // cudaPol(res, [e] ZS_LAMBDA(auto &v) { v = e; });
    cudaPol(range(verts.size()), [res = proxy<space>(res), verts = proxy<space>({}, verts), offset, nwarps, nchn, top,
                                  rop] ZS_LAMBDA(int i) mutable {
        auto [mask, numValid] = warp_mask(i, nwarps);
        auto locid = threadIdx.x & 31;
        float v = top(verts(offset, i));
        while (--nchn) {
            v = rop(top(verts(offset++, i)), v);
        }
        reduce_to(i, nwarps, v, res[i / 32], rop);
    });

    Vector<float> ret{res.get_allocator(), 1};
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), e, rop);
    return ret.getVal();
}

struct ZSPrimitiveReduction : zeno::INode {
    struct pass_on {
        constexpr auto operator()(auto v) const noexcept {
            return v;
        }
    };
    struct getabs {
        constexpr auto operator()(auto v) const noexcept {
            return zs::abs(v);
        }
    };
    virtual void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto prim = get_input<ZenoParticles>("ZSParticles");
        auto &verts = prim->getParticles();
        auto attrToReduce = get_input2<std::string>("attr");
        if (attrToReduce == "pos")
            attrToReduce = "x";
        if (attrToReduce == "vel")
            attrToReduce = "v";

        if (!verts.hasProperty(attrToReduce))
            throw std::runtime_error(fmt::format("verts do not have property [{}]\n", attrToReduce));

        auto opStr = get_input2<std::string>("op");
        zeno::NumericValue result;
        if (opStr == "avg") {
            result = prim_reduce(verts, 0, pass_on{}, std::plus<float>{}, attrToReduce) / verts.size();
        } else if (opStr == "max") {
            result = prim_reduce(verts, limits<float>::lowest(), pass_on{}, getmax<float>{}, attrToReduce);
        } else if (opStr == "min") {
            result = prim_reduce(verts, limits<float>::max(), pass_on{}, getmin<float>{}, attrToReduce);
        } else if (opStr == "absmax") {
            result = prim_reduce(verts, 0, getabs{}, getmax<float>{}, attrToReduce);
        }

        auto out = std::make_shared<zeno::NumericObject>();
        out->set(result);
        set_output("result", std::move(out));
    }
};
ZENDEFNODE(ZSPrimitiveReduction, {/* inputs: */ {
                                      "ZSParticles",
                                      {"string", "attr", "pos"},
                                      {"enum avg max min absmax", "op", "avg"},
                                  },
                                  /* outputs: */
                                  {
                                      "result",
                                  },
                                  /* params: */
                                  {},
                                  /* category: */
                                  {
                                      "primitive",
                                  }});

} // namespace zeno