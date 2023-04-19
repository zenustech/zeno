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

struct zs_erode_smooth_flow : INode {
    void apply() override {
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
        auto smooth_rate = get_input2<float>("smoothRate");
        auto flowName = get_input2<std::string>("flowName");
        // 初始化网格属性
        auto &flow = terrain->verts.attr<float>(flowName);
        auto &_lap = terrain->verts.add_attr<float>("_lap");
        ////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////
        // 计算
        ////////////////////////////////////////////////////////////////////////////////////////
        /// @brief  accelerate cond computation using cuda
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        auto pol = cuda_exec();
        zs::Vector<float> zsflow{flow.size(), memsrc_e::device, 0};
        zs::Vector<float> zslap{_lap.size(), memsrc_e::device, 0};
        /// @brief  copy host-side attribute [flow]
        Resource::copy(MemoryEntity{zsflow.memoryLocation(), (void *)zsflow.data()},
                       MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)flow.data()},
                       flow.size() * sizeof(float));
        ///
        pol(range((std::size_t)nz * (std::size_t)nx),
            [flow = view<space>(zsflow), lap = view<space>(zslap), nx, nz] __device__(std::size_t idx) mutable {
                auto id_z = idx / nx; // outer index
                auto id_x = idx % nx; // inner index
                float net_diff = 0.0f;
                net_diff += flow[idx - 1 * (id_x > 0)];
                net_diff += flow[idx + 1 * (id_x < nx - 1)];
                net_diff += flow[idx - nx * (id_z > 0)];
                net_diff += flow[idx + nx * (id_z < nz - 1)];
                net_diff *= 0.25f;
                net_diff -= flow[idx];
                lap[idx] = net_diff;
            });

        pol(range((std::size_t)nz * (std::size_t)nx), [flow = view<space>(zsflow), lap = view<space>(zslap),
                                                       smooth_rate, nx, nz] __device__(std::size_t idx) mutable {
            auto id_z = idx / nx; // outer index
            auto id_x = idx % nx; // inner index
            float net_diff = 0.0f;
            net_diff += lap[idx - 1 * (id_x > 0)];
            net_diff += lap[idx + 1 * (id_x < nx - 1)];
            net_diff += lap[idx - nx * (id_z > 0)];
            net_diff += lap[idx + nx * (id_z < nz - 1)];
            net_diff *= 0.25f;
            net_diff -= lap[idx];
            flow[idx] -= smooth_rate * 0.5f * net_diff;
        });
        /// @brief  write back to host-side attribute
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)flow.data()},
                       MemoryEntity{zsflow.memoryLocation(), (void *)zsflow.data()}, zsflow.size() * sizeof(float));
        Resource::copy(MemoryEntity{MemoryLocation{memsrc_e::host, -1}, (void *)_lap.data()},
                       MemoryEntity{zslap.memoryLocation(), (void *)zslap.data()}, zslap.size() * sizeof(float));

        terrain->verts.erase_attr("_lap");
        set_output("prim_2DGrid", std::move(terrain));
    }
};

ZENDEFNODE(zs_erode_smooth_flow, {/* inputs: */ {
                                      "prim_2DGrid",
                                      {"float", "smoothRate", "1.0"},
                                      {"string", "flowName", "flow"},
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