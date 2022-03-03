#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {


struct MakePrimitive : zeno::INode {
  virtual void apply() override {
    auto prim = std::make_shared<PrimitiveObject>();
    auto size = get_input<NumericObject>("size")->get<int>();
    prim->resize(size);
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(MakePrimitive,
    { /* inputs: */ {
    {"int", "size", ""},
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

struct PrimitiveResize : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto size = get_input<NumericObject>("size")->get<int>();
    prim->resize(size);

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveResize,
    { /* inputs: */ {
    "prim",
    "size",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveGetSize : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto size = std::make_shared<NumericObject>();
    size->set<int>(prim->size());
    set_output("size", std::move(size));
  }
};

ZENDEFNODE(PrimitiveGetSize,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "size",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveGetFaceCount : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    auto size = std::make_shared<NumericObject>();
    size->set<int>(prim->tris.size() + prim->quads.size());
    set_output("size", std::move(size));
  }
};

ZENDEFNODE(PrimitiveGetFaceCount,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "size",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


}
