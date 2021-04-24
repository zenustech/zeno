#include <zen/zen.h>
#include <zen/ParticlesObject.h>
#include <zen/ShaderObject.h>
#include <Hg/IOUtils.h>

namespace zenbase {

struct ShadeParticles : zen::INode {
  virtual void apply() override {
    auto shad = zen::IObject::make<zenbase::ShaderObject>();

    const std::string basepath = "/home/bate/Develop/zensim/zenvis/";
    shad->vert = hg::file_get_content(basepath + "particles.vert");
    shad->frag = hg::file_get_content(basepath + "particles.frag");

    set_output("shader", shad);
  }
};

static int defShadeParticles = zen::defNodeClass<ShadeParticles>("ShadeParticles",
    { /* inputs: */ {
    }, /* outputs: */ {
        "shader",
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
