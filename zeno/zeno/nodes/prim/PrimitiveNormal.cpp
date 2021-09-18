#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct PrimitiveCalcNormal : zeno::INode {
  virtual void apply() override {
    auto prim = get_input<PrimitiveObject>("prim");

    auto &nrm = prim->add_attr<zeno::vec3f>("nrm");
    auto &pos = prim->attr<zeno::vec3f>("pos");
    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = zeno::vec3f(0);
    }

    for (size_t i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        auto n = zeno::cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);
        nrm[ind[0]] += n;
        nrm[ind[1]] += n;
        nrm[ind[2]] += n;
    }
    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = zeno::normalize(nrm[i]);
    }

    set_output("prim", get_input("prim"));
  }
};

ZENDEFNODE(PrimitiveCalcNormal, {
    {"prim"},
    {"prim"},
    {},
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
