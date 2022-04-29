#include "zeno/types/StringObject.h"
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
    auto type = get_input<StringObject>("type")->value;
    
    if(type=="edges" || type=="all")
      prim->lines.clear();
    if(type=="faces" || type=="all"){
      prim->tris.clear();
      prim->quads.clear();
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveClearConnect,
    { /* inputs: */ {
    "prim", {"enum edges faces all","type", "all"}
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
        prim->lines.reserve(n);
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


struct PrimitiveSplitEdges : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

    prim->foreach_attr([&] (auto &, auto &arr) {
        auto oldarr = arr;
        arr.resize(prim->tris.size() * 3);
        for (size_t i = 0; i < prim->tris.size(); i++) {
            auto ind = prim->tris[i];
            arr[i * 3 + 0] = oldarr[ind[0]];
            arr[i * 3 + 1] = oldarr[ind[1]];
            arr[i * 3 + 2] = oldarr[ind[2]];
        }
    });
    prim->resize(prim->tris.size() * 3);

    for (size_t i = 0; i < prim->tris.size(); i++) {
        prim->tris[i] = zeno::vec3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveSplitEdges, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});


struct PrimitiveFaceToEdges : zeno::INode {
  std::pair<int, int> sorted(int x, int y) {
      return x < y ? std::make_pair(x, y) : std::make_pair(y, x);
  }

  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");
    std::set<std::pair<int, int>> lines;

    for (int i = 0; i < prim->tris.size(); i++) {
        auto uvw = prim->tris[i];
        int u = uvw[0], v = uvw[1], w = uvw[2];
        lines.insert(sorted(u, v));
        lines.insert(sorted(v, w));
        lines.insert(sorted(u, w));
    }
    for (auto [u, v]: lines) {
        prim->lines.emplace_back(u, v);
    }

    if (get_param<bool>("clearFaces")) {
        prim->tris.clear();
    }
    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveFaceToEdges,
    { /* inputs: */ {
    "prim",
    }, /* outputs: */ {
    "prim",
    }, /* params: */ {
    {"bool", "clearFaces", "1"},
    }, /* category: */ {
    "primitive",
    }});



}
