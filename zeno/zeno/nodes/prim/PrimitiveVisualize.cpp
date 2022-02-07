#if 0
#ifdef ZENO_VISUALIZATION
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/Visualization.h>
#include <zeno/types/PrimitiveIO.h>
#include <zeno/utils/filesystem.h>

namespace zeno {

struct ToVisualize_PrimitiveObject : zeno::INode {
    virtual void apply() override {
        auto that = get_input<PrimitiveObject>("prim");
        auto path = get_param<std::string>("path");
        writezpm(that.get(), (path + ".zpm").c_str());
    }
};

ZENO_DEFOVERLOADNODE(ToVisualize, _PrimitiveObject, typeid(PrimitiveObject).name())({
        {"prim"},
        {},
        {{"string", "path", ""}},
        {"primitive"},
});


#if 0
struct PrimitiveShadeObject : zeno::IObject {
    std::string vertpath, fragpath;
    std::shared_ptr<PrimitiveObject> prim;
    std::string primtype;
};

struct ToVisualize_PrimitiveShadeObject : zeno::INode {
    virtual void apply() override {
        auto that = get_input<PrimitiveShadeObject>("prim");
        auto path = get_param<std::string>("path");
        fs::copy_file(that->vertpath, path + ".zpm." + that->primtype + ".vert");
        fs::copy_file(that->fragpath, path + ".zpm." + that->primtype + ".frag");

        if (auto node = graph->getOverloadNode("ToVisualize", {that}); node) {
            node->inputs["path:"] = path;
            node->doApply();
        }
    }
};

ZENO_DEFOVERLOADNODE(ToVisualize, _PrimitiveShadeObject, typeid(PrimitiveShadeObject).name())({
        {"prim"},
        {},
        {{"string", "path", ""}},
        {"primitive"},
});


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

}
#endif
#endif
