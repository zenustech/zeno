#ifdef ZENO_VISUALIZATION
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/Visualization.h>
#include <zeno/types/PrimitiveIO.h>
#include <zeno/utils/filesystem.h>

namespace zeno {

void PrimitiveObject_dumpfile(UserData &ud, PrimitiveObject *that) {
    auto path = ud.get<std::string>("path");
    writezpm(that, (path + ".zpm").c_str());
}

static int defPrimitiveObject_dumpfile = registerObjectMethod(
        "dumpfile", PrimitiveObject_dumpfile, {typeid(PrimitiveObject)});


struct PrimitiveShadeObject : zeno::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;
    std::string primtype;
};

void PrimitiveShadeObject_dumpfile(UserData &ud, PrimitiveShadeObject *that) {
    auto path = ud.get<std::string>("path");
    fs::copy_file(that->vertpath, path + ".zpm." + that->primtype + ".vert");
    fs::copy_file(that->fragpath, path + ".zpm." + that->primtype + ".frag");
    PrimitiveObject_dumpfile(that, ud);
}

static int defPrimitiveShadeObject_dumpfile = registerObjectMethod(
        "dumpfile", PrimitiveShadeObject_dumpfile, {typeid(PrimitiveShadeObject)});


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
    {"enum points lines tris quads", "primtype", "points"},
    {"readpath", "vertpath", "assets/particles.vert"},
    {"readpath", "fragpath", "assets/particles.frag"},
    }, /* category: */ {
    "visualize",
    }});

}
#endif
