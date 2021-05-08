#include <zen/zen.h>
#include <zen/ShaderObject.h>
#include <Hg/IOUtils.h>
#include <Hg/Archive.hpp>
#include <Hg/StrUtils.h>
#include "ShaderMacros.h"

namespace zenbase {

struct ShadeParticles : zen::INode {
  virtual void apply() override {
    auto shad = zen::IObject::make<zenbase::ShaderObject>();

    shad->vert = hg::Archive::getString("particles.vert");
    shad->frag = hg::Archive::getString("particles.frag");

    auto point_size = std::get<int>(get_param("point_size"));
    auto vel_mag = std::get<float>(get_param("vel_mag"));

    ShaderMacros macros;
    macros.add("D_POINT_SIZE", hg::StringBuilder() << point_size);
    macros.add("D_VEL_MAG", hg::StringBuilder() << vel_mag);

    macros.apply(shad->vert);
    macros.apply(shad->frag);

    set_output("shader", shad);
  }
};

static int defShadeParticles = zen::defNodeClass<ShadeParticles>("ShadeParticles",
    { /* inputs: */ {
    }, /* outputs: */ {
        "shader",
    }, /* params: */ {
        {"int", "point_size", "5 0"},
        {"float", "vel_mag", "10.0 0"},
    }, /* category: */ {
        "visualize",
    }});

}
