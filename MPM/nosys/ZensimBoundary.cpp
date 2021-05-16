#include "../ZenoSimulation.h"
#include "../ZensimGeometry.h"

namespace zenbase {

struct ZensimBoundary : zen::INode {
  void apply() override {
    auto boundary = zen::IObject::make<ZenoBoundary>();

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
    set_output("ZSBoundary", boundary);
  }
};

static int defZensimBoundary = zen::defNodeClass<ZensimBoundary>(
    "ZensimBoundary", {/* inputs: */ {"ZSLevelSet"},
                       /* outputs: */ {"ZSBoundary"},
                       /* params: */ {{"string", "type", "sticky"}},
                       /* category: */ {"simulation"}});

} // namespace zenbase