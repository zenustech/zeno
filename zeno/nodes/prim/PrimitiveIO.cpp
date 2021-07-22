#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/PrimitiveIO.h>
#include <zeno/StringObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace zeno {


struct ExportPrimitive : zeno::INode {
  virtual void apply() override {
    auto path = get_input<StringObject>("path")->get();
    auto prim = get_input<PrimitiveObject>("prim");
    writezpm(prim.get(), path.c_str());
  }
};

static int defExportPrimitive = zeno::defNodeClass<ExportPrimitive>("ExportPrimitive",
    { /* inputs: */ {
    "prim",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct ImportPrimitive : zeno::INode {
  virtual void apply() override {
    auto path = get_input<StringObject>("path");
    auto prim = std::make_shared<PrimitiveObject>();
    readzpm(prim.get(), path->get().c_str());
    set_output("prim", prim);
  }
};

static int defImportPrimitive = zeno::defNodeClass<ImportPrimitive>("ImportPrimitive",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
