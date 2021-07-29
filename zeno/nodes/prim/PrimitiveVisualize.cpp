#ifndef ZENO_VISUALIZATION
#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>

namespace zeno {

// TODO: why taint IObject with visualization stuffs?
ZENO_API void PrimitiveObject::dumpfile(std::string const &path) {
}

}
#else
#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/Visualization.h>
#include <zeno/PrimitiveIO.h>
#include <zeno/filesystem.h>

namespace zeno {

ZENO_API void PrimitiveObject::dumpfile(std::string const &path) {
    writezpm(this, (path + ".zpm").c_str());
}


struct PrimitiveShadeObject : zeno::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;
    std::string primtype;

    virtual void dumpfile(std::string const &path) override {
        fs::copy_file(vertpath, path + ".zpm." + primtype + ".vert");
        fs::copy_file(fragpath, path + ".zpm." + primtype + ".frag");
        prim->dumpfile(path);
    }
};


struct PrimitiveShade : zeno::INode {
  virtual void apply() override {
    auto shade = std::make_shared<PrimitiveShadeObject>();
    shade->prim = get_input<PrimitiveObject>("prim");
    shade->vertpath = get_param<std::string>("vertpath");
    shade->fragpath = get_param<std::string>("fragpath");
    shade->primtype = get_param<std::string>("primtype");

    set_output("shade", std::move(shade));
  }
};

ZENDEFNODE(PrimitiveShade,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "shade",
    }, /* params: */ {
    {"string", "primtype", "points"},
    {"string", "vertpath", "assets/particles.vert"},
    {"string", "fragpath", "assets/particles.frag"},
    }, /* category: */ {
    "visualize",
    }});

}
#endif
