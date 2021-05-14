#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/grid/GridOp.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/simulation/sparsity/SparsityCompute.hpp"

namespace zenbase {

struct GridUpdate : zen::INode {
  void apply() override {
#if 0
    // auto dt = get_input("dt")->as<zenbase::NumericObject>()->get<float>();
    auto maxVelSqr = zen::IObject::make<zenbase::NumericObject>();

    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    auto stepDt = std::get<float>(get_param("dt"));
    auto gravity = std::get<float>(get_param("gravity"));

    using GridT = zs::remove_cvref_t<decltype(grid)>;

    zs::Vector<float> velSqr{1, zs::memspace_e::um, 0};
    velSqr[0] = 0;
    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol(
        {(std::size_t)partition.size(), (std::size_t)Grid::block_t::space},
        zs::ComputeGridBlockVelocity{zs::wrapv<zs::execspace_e::cuda>{},
                                     zs::wrapv<zs::transfer_scheme_e::apic>{},
                                     grid, stepDt, gravity, velSqr.data()});

    maxVelSqr->get<float>() = velSqr[0];
    set_output("MaxVelSqr", maxVelSqr);
#endif
  }
};

static int defGridUpdate = zen::defNodeClass<GridUpdate>(
    "GridUpdate", {/* inputs: */ {"ZSPartition", "ZSGrid"},
                   /* outputs: */ {"MaxVelSqr"},
                   /* params: */
                   {{"float", "dt", "1"}, {"float", "gravity", "-9.8"}},
                   /* category: */ {"simulation"}});

struct ResolveBoundaryOnGrid : zen::INode {
  void apply() override {
#if 0
    // auto maxVelSqr = zen::IObject::make<ZenoGrid>();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto dx = std::get<float>(get_param("dx"));

    auto cudaPol = zs::cuda_exec().device(0);

    set_output("MaxVelSqr", maxVelSqr);
#endif
  }
};

static int defResolveBoundaryOnGrid = zen::defNodeClass<ResolveBoundaryOnGrid>(
    "ResolveBoundaryOnGrid",
    {/* inputs: */ {"ZSPartition", "ZSGrid", "ZSBoundary"},
     /* outputs: */ {},
     /* params: */ {{"float", "dx", "1"}},
     /* category: */ {"simulation"}});

} // namespace zenbase