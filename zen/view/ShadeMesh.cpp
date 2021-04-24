#include <zen/zen.h>
#include <zen/ShaderObject.h>
#include <Hg/IOUtils.h>
#include <Hg/StrUtils.h>

namespace zenbase {

struct ShaderMacros {
  std::string lines;

  void add(std::string const &name, std::string const &value) {
    lines += "#define " + name + " " + value + "\n";
  }

  void apply(std::string &source) {
    source = "#version 330 core\n" + lines + "/**************/\n" + source;
  }
};

struct ShadeMesh : zen::INode {
  virtual void apply() override {
    auto shad = zen::IObject::make<zenbase::ShaderObject>();

    const std::string basepath = "/home/bate/Develop/zensim/assets/";
    shad->vert = hg::file_get_content(basepath + "mesh.vert");
    shad->frag = hg::file_get_content(basepath + "mesh.frag");

    auto color_r = std::get<float>(get_param("color_r"));
    auto color_g = std::get<float>(get_param("color_g"));
    auto color_b = std::get<float>(get_param("color_b"));
    auto roughness = std::get<float>(get_param("roughness"));
    auto metallic = std::get<float>(get_param("metallic"));

    ShaderMacros macros;
    macros.add("D_ALBEDO", hg::StringBuilder() << "vec3(" << color_r << ", "
                            << color_g << ", " << color_b << ")");
    macros.add("D_ROUGHNESS", hg::StringBuilder() << roughness);
    macros.add("D_METALLIC", hg::StringBuilder() << metallic);

    macros.apply(shad->vert);
    macros.apply(shad->frag);

    set_output("shader", shad);
  }
};

static int defShadeMesh = zen::defNodeClass<ShadeMesh>("ShadeMesh",
    { /* inputs: */ {
    }, /* outputs: */ {
        "shader",
    }, /* params: */ {
        {"float", "color_r", "0.8 0 1"},
        {"float", "color_g", "0.8 0 1"},
        {"float", "color_b", "0.8 0 1"},
        {"float", "roughness", "0.4 0 1"},
        {"float", "metallic", "0.0 0 1"},
    }, /* category: */ {
        "visualize",
    }});

}
