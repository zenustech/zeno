#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <cassert>

namespace zeno {


struct MakePrimitive : zeno::INode {
  virtual void apply() override {
    auto prim = std::make_shared<PrimitiveObject>();
    int size = get_input<NumericObject>("size")->get<int>();
    if (size == 0) {
        auto points = get_input<StringObject>("points")->get();
        std::string num;
        zeno::vec3f vert;
        int idx = 0;
        for (auto c : points) {
            if (c == ' ') {
                vert[idx++] = std::stof(num);
                if (idx == 3) {
                    idx = 0;
                    prim->verts.push_back(vert);
                }
                num = "";
            }
            else
                num.push_back(c);
        }
    }
    else {
        prim->resize(size);
    }
    set_output("prim", std::move(prim));
  }
};

ZENDEFNODE(MakePrimitive,
    { /* inputs: */ {
    {"int", "size", "0"},
    {"string", "points", ""},
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
    {"int","size"},
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
    {"int","size"},
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
        {"int","size"},
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


}
