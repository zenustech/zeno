#include "../ZenoSimulation.h"
#include "../ZensimGeometry.h"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zeno {

struct ZensimBoundary : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ZensinBoundary\n");
    auto boundary = zeno::IObject::make<ZenoBoundary>();

    auto type = std::get<std::string>(get_param("type"));
    auto queryType = [&type]() -> zs::collider_e {
      if (type == "sticky")
        return zs::collider_e::Sticky;
      else if (type == "slip")
        return zs::collider_e::Slip;
      else if (type == "separate")
        return zs::collider_e::Separate;
      return zs::collider_e::Sticky;
    };
    // pass in FloatGrid::Ptr
    auto &ls = get_input("ZSLevelSet")->as<ZenoSparseLevelSet>()->get();

    boundary->get() = zs::LevelSetBoundary{ls, queryType()};
    fmt::print(fg(fmt::color::cyan), "done executing ZensinBoundary\n");
    set_output("ZSBoundary", boundary);
  }
};

static int defZensimBoundary = zeno::defNodeClass<ZensimBoundary>(
    "ZensimBoundary", {/* inputs: */ {"ZSLevelSet"},
                       /* outputs: */ {"ZSBoundary"},
                       /* params: */ {{"string", "type", "sticky"}},
                       /* category: */ {"GPUMPM"}});

} // namespace zeno