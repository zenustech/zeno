#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ZenoInc.h>
#include <openvdb/tools/LevelSetSphere.h>

namespace zeno {

struct VDBCreateLevelsetSphere : zeno::INode {
  virtual void apply() override {
    //auto dx = std::get<float>(get_param("dx"));
    float dx=0.08f;
    if(has_input("Dx"))
    {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    float radius=1.0f;
    if(has_input("radius"))
    {
      radius = get_input("radius")->as<NumericObject>()->get<float>();
    }
    vec3f center(0);
    if(has_input("center"))
    {
      center = get_input("center")->as<NumericObject>()->get<vec3f>();
    }
    float half_width=(float)openvdb::LEVEL_SET_HALF_WIDTH;
    if(has_input("half_width"))
    {
      if (auto t = get_input("half_width")->as<NumericObject>()->get<float>(); t > 0)
        half_width=t;
    }
    auto data = std::make_shared<VDBFloatGrid>(openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, openvdb::Vec3f(center[0], center[1], center[2]), dx, half_width));
    set_output("data", data);
  }
};

static int defVDBCreateLevelsetSphere = zeno::defNodeClass<VDBCreateLevelsetSphere>(
    "VDBCreateLevelsetSphere", {/* inputs: */ {{"float","Dx","0.08"},{"float","radius","1.0"},{"vec3f","center","0,0,0"},{"float","half_width","3.0"}}, /* outputs: */
                    {
                        "data",
                    },
                    /* params: */
                    {
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

}
