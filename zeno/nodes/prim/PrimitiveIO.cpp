#include <zeno/zen.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/PrimitiveIO.h>
#include <zeno/StringObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace zen {


struct ExportPrimitive : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto prim = get_input("prim")->as<PrimitiveObject>();
    writezpm(prim, path->get().c_str());
  }
};

static int defExportPrimitive = zen::defNodeClass<ExportPrimitive>("ExportPrimitive",
    { /* inputs: */ {
    "prim",
    "path",
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct ImportPrimitive : zen::INode {
  virtual void apply() override {
    auto path = get_input("path")->as<StringObject>();
    auto prim = zen::IObject::make<PrimitiveObject>();
    readzpm(prim.get(), path->get().c_str());
    set_output("prim", prim);
  }
};

static int defImportPrimitive = zen::defNodeClass<ImportPrimitive>("ImportPrimitive",
    { /* inputs: */ {
    "path",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
