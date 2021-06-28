#include <zen/zen.h>
#include <zen/PrimitiveObject.h>
#include <zen/Visualization.h>
#include <zen/PrimitiveIO.h>
#include <zen/filesystem.h>

namespace zen {

ZENAPI void PrimitiveObject::visualize() {
    auto path = Visualization::exportPath("zpm");
    writezpm(this, path.c_str());
}


struct PrimitiveShadeObject : zen::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;

    virtual void visualize() override {
        auto path = Visualization::exportPath("zpm");
        fs::copy_file(vertpath, path + ".vert");
        fs::copy_file(fragpath, path + ".frag");
        writezpm(this, path.c_str());
    }
};


struct PrimitiveShade : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto shade = std::make_shared<PrimitiveShadeObject>();
    shade->prim = std::move(prim);
    shade->vertpath = get_param<std::string>("vertpath");
    shade->fragpath = get_param<std::string>("fragpath");

    set_output_ref("primShade", std::move(shade));
  }
};

ZENDEF(PrimitiveShade,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "primShade",
    }, /* params: */ {
    {"string", "vertpath", "xxx.vert"},
    {"string", "fragpath", "xxx.frag"},
    }, /* category: */ {
    "primitive",
    }});

}
