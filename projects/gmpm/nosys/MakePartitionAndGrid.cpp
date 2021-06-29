#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimObject.h"

namespace zen {

struct MakePartition : zen::INode {
  void apply() override {
    auto cnt = std::get<int>(get_param("count"));
    zs::HashTable<zs::i32, 3, int> ret{cnt, zs::memsrc_e::um, 0};

    auto partition = zen::IObject::make<ZenoPartition>();
    partition->get() = std::move(ret);
    set_output("ZSPartition", partition);
  }
};

static int defMakePartition = zen::defNodeClass<MakePartition>(
    "MakePartition", {/* inputs: */ {""},
                      /* outputs: */ {"ZSPartition"},
                      /* params: */ {{"int", "count", "1"}},
                      /* category: */ {"GPUMPM"}});

struct MakeGrid : zen::INode {
  void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    auto cnt = std::get<int>(get_param("count"));
    using GridT = zs::GridBlocks<zs::GridBlock<zs::dat32, 3, 2, 2>>;
    GridT gridblocks{dx, cnt, zs::memsrc_e::um, 0};

    auto grid = zen::IObject::make<ZenoGrid>();
    grid->get() = std::move(gridblocks);
    set_output("ZSGrid", grid);
  }
};

static int defMakeGrid = zen::defNodeClass<MakeGrid>(
    "MakeGrid", {/* inputs: */ {""},
                 /* outputs: */ {"ZSGrid"},
                 /* params: */ {{"float", "dx", "1"}, {"int", "count", "1"}},
                 /* category: */ {"GPUMPM"}});

} // namespace zen