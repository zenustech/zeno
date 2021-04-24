#include <zen/zen.h>
#include <zen/ShaderObject.h>
#include <Hg/IOUtils.h>

namespace zenbase {

struct ShadeMesh : zen::INode {
  virtual void apply() override {
    auto shad = zen::IObject::make<zenbase::ShaderObject>();

    const std::string basepath = "/home/bate/Develop/zensim/zenvis/";
    shad->vert = hg::file_get_content(basepath + "mesh.vert");
    shad->frag = hg::file_get_content(basepath + "mesh.frag");

    set_output("shader", shad);
  }
};

static int defShadeMesh = zen::defNodeClass<ShadeMesh>("ShadeMesh",
    { /* inputs: */ {
    }, /* outputs: */ {
        "shader",
    }, /* params: */ {
    }, /* category: */ {
        "visualize",
    }});

}
