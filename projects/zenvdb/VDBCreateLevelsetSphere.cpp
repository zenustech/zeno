#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
#include <zeno/StringObject.h>
#include <zeno/NumericObject.h>
#include <zeno/ZenoInc.h>
#include <openvdb/tools/LevelSetSphere.h>

namespace zeno {

struct VDBCreateLevelsetSphere : zeno::INode {
  virtual void apply() override {
    //auto dx = get_param_float("dx");
    float dx=0.08f;
    if(has_input("Dx"))
    {
      dx = get_input2_float("Dx");
    }
    float radius=1.0f;
    if(has_input("radius"))
    {
      radius = get_input2_float("radius");
    }
    vec3f center(0);
    if(has_input("center"))
    {
      center = toVec3f(get_input2_vec3f("center"));
    }
    float half_width=(float)openvdb::LEVEL_SET_HALF_WIDTH;
    if(has_input("half_width"))
    {
      if (auto t = get_input2_float("half_width"); t > 0)
        half_width=t;
    }
    auto data = std::make_unique<VDBFloatGrid>(openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, openvdb::Vec3f(center[0], center[1], center[2]), dx, half_width));
    set_output("data", std::move(data));
  }
};

static int defVDBCreateLevelsetSphere = zeno::defNodeClass<VDBCreateLevelsetSphere>(
    "VDBCreateLevelsetSphere", {/* inputs: */ {{gParamType_Float,"Dx","0.08"},{gParamType_Float,"radius","1.0"},{gParamType_Vec3f,"center","0,0,0"},{gParamType_Float,"half_width","3.0"}}, /* outputs: */
                    {
                        {gParamType_VDBGrid, "data"},
                    },
                    /* params: */
                    {
                    },
                    /* category: */
                    {
                        "openvdb",
                    }});

}
