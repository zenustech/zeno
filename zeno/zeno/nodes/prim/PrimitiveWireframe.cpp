#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <random>
#include <cmath>
#include <set>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

struct PrimitiveWireframe : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        std::set<std::pair<int, int>> segments;
        auto append = [&] (int i, int j) {
            segments.emplace(std::min(i, j), std::max(i, j));
        };
        for (auto const &ind: prim->lines) {
            append(ind[0], ind[1]);
        }
        for (auto const &ind: prim->tris) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[0]);
        }
        for (auto const &ind: prim->quads) {
            append(ind[0], ind[1]);
            append(ind[1], ind[2]);
            append(ind[2], ind[3]);
            append(ind[3], ind[0]);
        }
        prim->lines.clear();
        prim->lines.reserve(segments.size());
        for (auto const &[i, j]: segments) {
            prim->lines.push_back({i, j});
        }
        if (get_param<bool>("removeFaces")) {
            prim->tris.clear();
            prim->quads.clear();
        }

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveWireframe, {
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    {"bool", "removeFaces", "1"},
    },
    {"primitive"},
});

}
