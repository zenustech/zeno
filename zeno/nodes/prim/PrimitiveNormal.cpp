#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zen {


struct PrimitiveCalcNormal : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();
    auto &nrm = prim->add_attr<zen::vec3f>("nrm");
    auto &pos = prim->attr<zen::vec3f>("pos");

    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = zen::vec3f(0);
    }

    for (size_t i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        auto n = zen::cross(pos[ind[1]] - pos[ind[0]], pos[ind[2]] - pos[ind[0]]);
        nrm[ind[0]] += n;
        nrm[ind[1]] += n;
        nrm[ind[2]] += n;
    }

    for (size_t i = 0; i < nrm.size(); i++) {
        nrm[i] = zen::normalize(nrm[i]);
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

ZENDEFNODE(PrimitiveCalcNormal, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});

struct PrimitiveSplitEdges : zen::INode {
  virtual void apply() override {
    auto prim = get_input("prim")->as<PrimitiveObject>();

    for (auto &[_, arr]: prim->m_attrs) {
        std::visit([&prim](auto &arr) {
            auto oldarr = arr;
            arr.resize(prim->tris.size() * 3);
            for (size_t i = 0; i < prim->tris.size(); i++) {
                auto ind = prim->tris[i];
                arr[i * 3 + 0] = oldarr[ind[0]];
                arr[i * 3 + 1] = oldarr[ind[1]];
                arr[i * 3 + 2] = oldarr[ind[2]];
            }
        }, arr);
    }
    prim->resize(prim->tris.size() * 3);

    for (size_t i = 0; i < prim->tris.size(); i++) {
        prim->tris[i] = zen::vec3i(i * 3 + 0, i * 3 + 1, i * 3 + 2);
    }

    set_output_ref("prim", get_input_ref("prim"));
  }
};

ZENDEFNODE(PrimitiveSplitEdges, {
    {"prim"},
    {"prim"},
    {},
    {"primitive"},
});

}
