#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/tuple_hash.h>
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

        std::set<vec2i, tuple_less> segments;
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
        for (auto const &[start, len]: prim->polys) {
            if (len < 2)
                continue;
            for (int i = start + 1; i < start + len; i++) {
                append(prim->loops[i - 1], prim->loops[i]);
            }
            append(prim->loops[start + len - 1], prim->loops[start]);
        }
        prim->lines.values.assign(segments.begin(), segments.end());
        if (get_param<bool>("removeFaces")) {
            prim->tris.clear();
            prim->quads.clear();
            prim->loops.clear();
            prim->polys.clear();
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
