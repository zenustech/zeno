#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/simulation/grid/GridOp.hpp"
#include "zensim/simulation/sparsity/SparsityCompute.hpp"
#include "zensim/simulation/sparsity/SparsityOp.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/NumericObject.h>
namespace zeno {

struct GridFromPartition : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing GridFromPartition\n");
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto mloc = zs::match(
        [](auto &partition) { return partition.memoryLocation(); })(partition);
    auto cnt =
        zs::match([](auto &partition) { return partition.size(); })(partition);
    auto dx = std::get<float>(get_param("dx"));
    if (has_input("Dx")) {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    // auto grid = zeno::IObject::make<ZenoGrid>();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    using GridT = zs::GridBlocks<zs::GridBlock<zs::dat32, 3, 2, 2>>;
    GridT &gridblocks = std::get<GridT>(grid);
    if (gridblocks.dx.asFloat() != dx)
      gridblocks = GridT{dx, (size_t)cnt, mloc.memspace(), mloc.devid()};
    else
      gridblocks.resize(cnt);

    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol(
        {(std::size_t)cnt, (std::size_t)GridT::block_t::space()},
        zs::CleanGridBlocks{zs::wrapv<zs::execspace_e::cuda>{}, gridblocks});
    // fmt::print("{} grid blocks from partition\n", cnt);

    // grid = gridblocks;
    fmt::print(fg(fmt::color::cyan), "done executing GridFromPartition\n");
    // set_output("ZSGrid", grid);
  }
};

static int defGridFromPartition = zeno::defNodeClass<GridFromPartition>(
    "GridFromPartition", {/* inputs: */ {"ZSGrid", "ZSPartition", "Dx"},
                          /* outputs: */ {},
                          /* params: */ {{"float", "dx", "1"}},
                          /* category: */ {"GPUMPM"}});

struct ExpandPartition : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ExpandPartition\n");
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto lo = std::get<int>(get_param("lo"));
    auto hi = std::get<int>(get_param("hi"));

    auto cudaPol = zs::cuda_exec().device(0);
    zs::match([&, lo, hi](auto &partition) {
      using Table = zs::remove_cvref_t<decltype(partition)>;
      cudaPol({partition.size()},
              zs::EnlargeSparsity{zs::wrapv<zs::execspace_e::cuda>{}, partition,
                                  Table::key_t ::uniform(lo),
                                  Table::key_t ::uniform(hi)});
      // fmt::print("expanded to {} grid blocks\n", partition.size());
    })(partition);
    fmt::print(fg(fmt::color::cyan), "done executing ExpandPartition\n");
  }
};

static int defExpandPartition = zeno::defNodeClass<ExpandPartition>(
    "ExpandPartition", {/* inputs: */ {"ZSPartition"},
                        /* outputs: */ {},
                        /* params: */ {{"int", "lo", "0"}, {"int", "hi", "1"}},
                        /* category: */ {"GPUMPM"}});

} // namespace zeno