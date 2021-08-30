#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveIO.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>

namespace zeno {


struct ExportZpmPrimitive : zeno::INode {
  virtual void apply() override {
    auto path = get_input<StringObject>("path")->get();
    auto prim = get_input<PrimitiveObject>("prim");
    writezpm(prim.get(), path.c_str());
  }
};

ZENDEFNODE(ExportZpmPrimitive,
    { /* inputs: */ {
    {"PrimitiveObject", "prim"},
    {"writepath", "path"},
    }, /* outputs: */ {
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct ImportZpmPrimitive : zeno::INode {
  virtual void apply() override {
    auto path = get_input<StringObject>("path");
    auto prim = std::make_shared<PrimitiveObject>();
    readzpm(prim.get(), path->get().c_str());
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(ImportZpmPrimitive,
    { /* inputs: */ {
    {"readpath", "path"},
    }, /* outputs: */ {
    {"PrimitiveObject", "prim"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

}
