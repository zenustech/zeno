#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct PrimitiveSimplePoints : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    size_t points_count = prim->size();
    prim->points.resize(points_count);
    for (int i = 0; i < points_count; i++) {
      prim->points[i] = i;
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSimplePoints,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleLines : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    size_t lines_count = prim->size() / 2;
    prim->lines.resize(lines_count);
    for (int i = 0; i < lines_count; i++) {
      prim->lines[i] = zeno::vec2i(2 * i, 2 * i + 1);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSimpleLines,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleTris : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    size_t tris_count = prim->size() / 3;
    prim->tris.resize(tris_count);
    for (int i = 0; i < tris_count; i++) {
      prim->tris[i] = zeno::vec3i(3 * i, 3 * i + 1, 3 * i + 2);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSimpleTris,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveSimpleQuads : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    size_t quads_count = prim->size() / 4;
    prim->quads.resize(quads_count);
    for (int i = 0; i < quads_count; i++) {
      prim->quads[i] = zeno::vec4i(4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSimpleQuads,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});


struct PrimitiveClearConnect : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    prim->points.clear();
    prim->lines.clear();
    prim->tris.clear();
    prim->quads.clear();

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveClearConnect,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    }, /* category: */ {
    "primitive",
    }});

struct PrimitiveLineSimpleLink : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");

        prim->lines.clear();
        intptr_t n = prim->verts.size();
        for (intptr_t i = 1; i < n; i++) {
            prim->lines.emplace_back(i - 1, i);
        }
        prim->lines.update();
        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveLineSimpleLink, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});



}
