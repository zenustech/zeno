#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/grid/GridOp.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/simulation/sparsity/SparsityCompute.hpp"

namespace zenbase {

struct GridFromPartition : zen::INode {
  void apply() override {
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto mh =
        zs::match([](auto &partition) { return partition.base(); })(partition);
    auto cnt =
        zs::match([](auto &partition) { return partition.size(); })(partition);
    auto dx = std::get<float>(get_param("dx"));

    auto grid = zen::IObject::make<ZenoGrid>();
    using GridT = zs::GridBlocks<zs::GridBlock<zs::dat32, 3, 2, 2>>;
    GridT gridblocks{dx, cnt, mh.memspace(), mh.devid()};

    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol(
        {(std::size_t)cnt, (std::size_t)GridT::block_t::space},
        zs::CleanGridBlocks{zs::wrapv<zs::execspace_e::cuda>{}, gridblocks});

    grid->get() = gridblocks;
    set_output("ZSGrid", grid);
  }
};

static int defGridFromPartition = zen::defNodeClass<GridFromPartition>(
    "GridFromPartition", {/* inputs: */ {"ZSPartition"},
                          /* outputs: */ {"ZSGrid"},
                          /* params: */ {{"float", "dx", "1"}},
                          /* category: */ {"simulation"}});

} // namespace zenbase