#if 0
#include <zen/zen.h>
#include <zen/VDBGrid.h>
//#include "../../Library/MnBase/Meta/Polymorphism.h"
//openvdb::io::File(filename).write({grid});

namespace zenbase {

struct SetVDBTransform : zen::INode {
  virtual void apply() override {
    auto dx = std::get<float>(get_param("dx"));
    auto grid = get_input("grid")->as<VDBGrid>();
    auto position = zen::get_float3<openvdb::Vec3f>(get_param("position"));
    auto rotation = zen::get_float3<openvdb::Vec3f>(get_param("rotation"));
    auto scale = zen::get_float3<openvdb::Vec3f>(get_param("scale"));

    auto transform = openvdb::math::Transform::createLinearTransform(dx);
    transform->postRotate(rotation[0], openvdb::math::X_AXIS);
    transform->postRotate(rotation[1], openvdb::math::Y_AXIS);
    transform->postRotate(rotation[2], openvdb::math::Z_AXIS);
    grid->setTransform(transform);
  }
};


static int defSetVDBTransform = zen::defNodeClass<SetVDBTransform>("SetVDBTransform",
    { /* inputs: */ {
    "grid",
    }, /* outputs: */ {
    }, /* params: */ {
    {"float", "dx", "0.08 0"},
    {"float3", "position", "0 0 0"},
    {"float3", "rotation", "0 0 0"},
    {"float3", "scale", "1 1 1"},
    }, /* category: */ {
    "openvdb",
    }});

}
#endif
