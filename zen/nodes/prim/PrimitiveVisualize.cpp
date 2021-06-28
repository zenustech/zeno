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


struct PrimitiveShadePointsObject : zen::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;

    virtual void visualize() override {
        auto path = Visualization::exportPath("zpm");
        fs::copy_file(vertpath, path + ".points.vert");
        fs::copy_file(fragpath, path + ".points.frag");
        writezpm(prim.get(), path.c_str());
    }
};


struct PrimitiveShadePoints : zen::INode {
  virtual void apply() override {
    auto shade = std::make_shared<PrimitiveShadePointsObject>();
    shade->prim = get_input<PrimitiveObject>("prim");
    shade->vertpath = get_param<std::string>("vertpath");
    shade->fragpath = get_param<std::string>("fragpath");

    set_output_ref("shade", std::move(shade));
  }
};

ZENDEFNODE(PrimitiveShadePoints,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "shade",
    }, /* params: */ {
    {"string", "vertpath", "xxx.vert"},
    {"string", "fragpath", "xxx.frag"},
    }, /* category: */ {
    "visualize",
    }});

}
