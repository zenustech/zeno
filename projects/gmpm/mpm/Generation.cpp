#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/geometry/VdbSampler.h"
#include <zeno/VDBGrid.h>

namespace zeno {

struct ToZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ToZSLevelSet\n");
    auto ls = IObject::make<ZenoLevelSet>();

    if (has_input<VDBFloatGrid>("VDBFloatGrid")) {
      // pass in FloatGrid::Ptr
      zs::OpenVDBStruct gridPtr =
          get_input<VDBFloatGrid>("VDBFloatGrid")->m_grid;
      ls->getLevelSet() = zs::convertFloatGridToSparseLevelSet(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0});
    } else {
      auto path = get_param<std::string>("path");
      auto gridPtr = zs::loadFloatGridFromVdbFile(path);
      ls->getLevelSet() = zs::convertFloatGridToSparseLevelSet(
          gridPtr, zs::MemoryProperty{zs::memsrc_e::um, 0});
    }

    fmt::print(fg(fmt::color::cyan), "done executing ToZSLevelSet\n");
    set_output("ZSLevelSet", std::move(ls));
  }
};
ZENDEFNODE(ToZSLevelSet, {
                             {"VDBFloatGrid", "VDBFloat3Grid"},
                             {"ZSLevelSet"},
                             {{"string", "path", ""}},
                             {"MPM"},
                         });

struct ZSLevelSetToVDBGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZSLevelSetToVDBGrid\n");
    auto vdb = IObject::make<VDBFloatGrid>();

    if (has_input<ZenoLevelSet>("ZSLevelSet")) {
      auto ls = get_input<ZenoLevelSet>("ZSLevelSet");
      if (ls->holdSparseLevelSet()) {
        vdb->m_grid =
            zs::convertSparseLevelSetToFloatGrid(ls->getSparseLevelSet())
                .as<openvdb::FloatGrid::Ptr>();
      } else
        ZS_WARN("The current input levelset is not a sparse levelset!");
    }

    fmt::print(fg(fmt::color::cyan), "done executing ZSLevelSetToVDBGrid\n");
    set_output("VDBFloatGrid", std::move(vdb));
  }
};
ZENDEFNODE(ZSLevelSetToVDBGrid, {
                                    {"ZSLevelSet"},
                                    {"VDBFloatGrid"},
                                    {{"string", "path", ""}},
                                    {"MPM"},
                                });

} // namespace zeno