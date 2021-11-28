#include "Structures.hpp"
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
                             {"VDBFloatGrid", "VDBVecGrid"},
                             {"ZSLevelSet"},
                             {{"string", "path", ""}},
                             {"MPM"},
                         });

} // namespace zeno