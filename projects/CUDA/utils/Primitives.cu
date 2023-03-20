#include "Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <zeno/types/DummyObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
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

template <typename T, typename Op>
__forceinline__ __device__ void reduce_to(int i, int n, T val, T &dst, Op op) {
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
float prim_reduce(typename ZenoParticles::particles_t &verts, float e, TransOp top, ReduceOp rop,
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
        template <typename T>
        constexpr T operator()(T v) const noexcept {
            return v;
        }
    };
    struct getabs {
        template <typename T>
        constexpr T operator()(T v) const noexcept {
            return zs::abs(v);
        }
    };
    virtual void apply() override {
        using namespace zs;
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

struct ZSGetUserData : zeno::INode {
    virtual void apply() override {
        auto object = get_input<ZenoParticles>("object");
        auto key = get_param<std::string>("key");
        auto hasValue = object->userData().has(key);
        auto data = hasValue ? object->userData().get(key) : std::make_shared<DummyObject>();
        set_output2("hasValue", hasValue);
        set_output("data", std::move(data));
    }
};

ZENDEFNODE(ZSGetUserData, {
                              {"object"},
                              {"data", {"bool", "hasValue"}},
                              {{"string", "key", ""}},
                              {"lifecycle"},
                          });

int Pos2Idx(const int x, const int z, const int nx) {
    return z * nx + x;
}
__forceinline__ __device__ unsigned int erode_random(float seed, int idx) {
    unsigned int s = *(unsigned int *)(&seed);
    s ^= idx << 3;
    s *= 179424691; // a magic prime number
    s ^= s << 13 | s >> (32 - 13);
    s ^= s >> 17 | s << (32 - 17);
    s ^= s << 23;
    s *= 179424691;
    return s;
}
// 降水/蒸发
struct zs_erode_value2cond : INode {
    void apply() override {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化
        ////////////////////////////////////////////////////////////////////////////////////////
        // 初始化网格
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        int nx, nz;
        auto &ud = terrain->userData();
        if ((!ud.has<int>("nx")) || (!ud.has<int>("nz")))
            zeno::log_error("no such UserData named '{}' and '{}'.", "nx", "nz");
        nx = ud.get2<int>("nx");
        nz = ud.get2<int>("nz");
        auto &pos = terrain->verts;
        float cellSize = std::abs(pos[0][0] - pos[1][0]);
        // 获取面板参数
        auto value = get_input2<float>("value");
        auto seed = get_input2<float>("seed");

        // 初始化网格属性
        if (!terrain->verts.has_attr("cond")) {
            auto &_cond = terrain->verts.add_attr<float>("cond");
            std::fill(_cond.begin(), _cond.end(), 0.0);
        }
        auto &attr_cond = terrain->verts.attr<float>("cond");
        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////

        /// @brief  accelerate cond computation using cuda
        auto pol = cuda_exec();
        zs::Vector<float> zscond{attr_cond.size(), memsrc_e::device, 0};
        pol(range((std::size_t)nz * (std::size_t)nx),
            [zscond = view<space>(zscond), value, seed, nx,
             nxnz = (std::size_t)nz * (std::size_t)nx] __device__(std::size_t idx) mutable {
                auto z = idx / nx; // outer index
                auto x = idx % nx; // inner index
                if (value >= 1.0f) {
                    zscond[idx] = 1;
                } else {
                    value = value < 0 ? 0 : (value > 1 ? 1 : value);
                    unsigned int cutoff = (unsigned int)(value * 4294967295.0);
                    unsigned int randval = erode_random(seed, idx + nxnz);
                    zscond[idx] = randval < cutoff;
                }
            });
        /// @brief  write back to host-side attribute
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)attr_cond.data()},
                       MemoryEntity{zscond.memoryLocation(), (void *)zscond.data()}, zscond.size() * sizeof(float));

        set_output("prim_2DGrid", std::move(terrain));
    }
};
ZENDEFNODE(zs_erode_value2cond, {/* inputs: */ {
                                     "prim_2DGrid",
                                     {"float", "value", "1.0"}, // 0.0 ~ 1.0
                                     {"float", "seed", "0.0"},
                                 },
                                 /* outputs: */
                                 {
                                     "prim_2DGrid",
                                 },
                                 /* params: */ {}, /* category: */
                                 {
                                     "erode",
                                 }});

} // namespace zeno