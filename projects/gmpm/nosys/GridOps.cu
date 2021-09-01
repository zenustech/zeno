#include "../ZenoSimulation.h"
#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/simulation/grid/GridOp.hpp"
#include "zensim/simulation/sparsity/SparsityCompute.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/core/INode.h>
#include <zeno/types/NumericObject.h>
#include <zeno/zeno.h>

namespace zeno {

struct GridUpdate : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing GridUpdate\n");
    // auto dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto maxVelSqr = zeno::IObject::make<zeno::NumericObject>();

    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    // auto stepDt = std::get<float>(get_param("dt"));
    auto stepDt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
    auto gravity = get_param<float>("gravity");
    auto extf = zs::vec<float, 3>::zeros();
    if (has_input("extforce")) {
      auto tmp = get_input<zeno::NumericObject>("extforce")->get<zeno::vec3f>();
      extf = zs::vec<float, 3>{tmp[0], tmp[1], tmp[2]};
    } else
      extf[1] = gravity;
    zs::Vector<float> velSqr{1, zs::memsrc_e::um, 0};
    velSqr[0] = 0;
    auto cudaPol = zs::cuda_exec().device(0);
    zs::match([&](auto &partition, auto &grid) {
      using GridT = zs::remove_cvref_t<decltype(grid)>;
      fmt::print("updating {} grid blocks\n", partition.size());
      cudaPol(
          {(std::size_t)partition.size(), (std::size_t)GridT::block_t::space()},
          zs::ComputeGridBlockVelocity{zs::wrapv<zs::execspace_e::cuda>{},
                                       zs::wrapv<zs::transfer_scheme_e::apic>{},
                                       grid, stepDt, extf, velSqr.data()});
    })(partition, grid);
    maxVelSqr->set<float>(velSqr[0]);
    fmt::print(fg(fmt::color::cyan), "done executing GridUpdate\n");
    set_output("MaxVelSqr", maxVelSqr);
  }
};

static int defGridUpdate = zeno::defNodeClass<GridUpdate>(
    "GridUpdate", {/* inputs: */ {"dt", "ZSPartition", "ZSGrid", "extforce"},
                   /* outputs: */ {"MaxVelSqr"},
                   /* params: */
                   {/*{"float", "dt", "1"}, */ {"float", "gravity", "-9.8"}},
                   /* category: */ {"GPUMPM"}});

struct ResolveBoundaryOnGrid : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ResolveBoundaryOnGrid\n");
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    auto &boundary = get_input("ZSBoundary")->as<ZenoBoundary>()->get();

    auto cudaPol = zs::cuda_exec().device(0);
    zs::match(
        [&](auto &collider, auto &partition, auto &grid)
            -> std::enable_if_t<
                zs::remove_cvref_t<decltype(collider)>::dim ==
                    zs::remove_cvref_t<decltype(partition)>::dim &&
                zs::remove_cvref_t<decltype(partition)>::dim ==
                    zs::remove_cvref_t<decltype(grid)>::dim> {
          // fmt::print("projecting {} grid blocks\n", partition.size());
          using Grid = zs::remove_cvref_t<decltype(grid)>;
          cudaPol({(std::size_t)partition.size(),
                   (std::size_t)Grid::block_t::space()},
                  zs::ApplyBoundaryConditionOnGridBlocks{
                      zs::wrapv<zs::execspace_e::cuda>{}, collider, partition,
                      grid});
        },
        [](...) {})(boundary, partition, grid);
    fmt::print(fg(fmt::color::cyan), "done executing ResolveBoundaryOnGrid\n");
  }
};

static int defResolveBoundaryOnGrid = zeno::defNodeClass<ResolveBoundaryOnGrid>(
    "ResolveBoundaryOnGrid",
    {/* inputs: */ {"ZSPartition", "ZSGrid", "ZSBoundary"},
     /* outputs: */ {},
     /* params: */ {},
     /* category: */ {"GPUMPM"}});

} // namespace zeno