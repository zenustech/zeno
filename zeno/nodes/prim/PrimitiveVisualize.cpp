#ifdef ZENO_VISUALIZATION
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/Visualization.h>
#include <zeno/types/PrimitiveIO.h>
#include <zeno/utils/filesystem.h>

namespace zeno {

void dumpfile_PrimitiveObject(UserData &ud, std::vector<IObject *> const &args) {
    auto that = static_cast<PrimitiveObject *>(args[0]);
    auto path = ud.get<std::string>("path");
    writezpm(that, (path + ".zpm").c_str());
}

static int def_dumpfile_PrimitiveObject = defObjectMethod(
        "dumpfile", dumpfile_PrimitiveObject,
        {typeid(PrimitiveObject).name()});


struct PrimitiveShadeObject : zeno::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;
    std::string primtype;
};

void ject_dumpfile_PrimitiveShadeObject(UserData &ud, std::vector<IObject *> const &args) {
    auto that = static_cast<PrimitiveShadeObject *>(args[0]);
    auto path = ud.get<std::string>("path");
    fs::copy_file(that->vertpath, path + ".zpm." + that->primtype + ".vert");
    fs::copy_file(that->fragpath, path + ".zpm." + that->primtype + ".frag");
    dumpfile_PrimitiveObject(ud, {that});
}

static int def_dumpfile_PrimitiveShadeObject = defObjectMethod(
        "dumpfile", dumpfile_PrimitiveShadeObject,
        {typeid(PrimitiveShadeObject).name()});


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
