#include "../ZensimGeometry.h"
#include "zensim/geometry/VdbLevelSet.h"
#include <zen/VDBGrid.h>

namespace zenbase {

struct ToZensimLevelSet : zen::INode {
  void apply() override {
    auto ls = zen::IObject::make<ZenoSparseLevelSet>();

    // auto path = std::get<std::string>(get_param("path"));
    // pass in FloatGrid::Ptr
    zs::OpenVDBStruct gridPtr =
        get_input("VDBFloatGrid")->as<VDBFloatGrid>()->m_grid;

    ls->get() = zs::convertFloatGridToSparseLevelSet(
        gridPtr, zs::MemoryHandle{zs::memsrc_e::um, 0});
    set_output("ZSLevelSet", ls);
  }
};

static int defToZensimLevelSet = zen::defNodeClass<ToZensimLevelSet>(
    "ToZensimLevelSet", {
                            /* inputs: */ {"VDBFloatGrid"},
                            /* outputs: */ {"ZSLevelSet"},
                            /* params: */ {},
                            /* category: */ {"simulation"}
                        });

} // namespace zenbase