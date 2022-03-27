#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>

namespace zeno {

ZENO_API void primPolygonate(PrimitiveObject *prim, bool with_uv) {
    prim->tris.reserve(prim->tris.size() + prim->polys.size());

    prim->loops.reserve(prim->loops.size() + prim->tris.size() * 3 + prim->quads.size() * 4);
    prim->polys.reserve(prim->polys.size() + prim->tris.size() + prim->quads.size());

    int old_loop_base = prim->loops.size();
    if (prim->tris.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->tris.size(); i++) {
            auto const &ind = prim->tris[i];
            prim->loops.push_back(ind[0]);
            prim->loops.push_back(ind[1]);
            prim->loops.push_back(ind[2]);
            prim->polys.push_back({base + i * 3, 3});
        }

        prim->tris.foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    if (prim->quads.size()) {
        int base = prim->loops.size();
        for (int i = 0; i < prim->quads.size(); i++) {
            auto const &ind = prim->quads[i];
            prim->loops.push_back(ind[0]);
            prim->loops.push_back(ind[1]);
            prim->loops.push_back(ind[2]);
            prim->loops.push_back(ind[3]);
            prim->polys.push_back({base + i * 4, 4});
        }

        prim->quads.foreach_attr([&] (auto const &key, auto const &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &newarr = prim->polys.add_attr<T>(key);
            newarr.insert(newarr.end(), arr.begin(), arr.end());
        });
    }

    prim->polys.update();

    if (!(!prim->tris.has_attr("uv0") || !prim->tris.has_attr("uv1") || !prim->tris.has_attr("uv2") || !with_uv)) {
        auto &uv0 = prim->tris.attr<vec3f>("uv0");
        auto &uv1 = prim->tris.attr<vec3f>("uv1");
        auto &uv2 = prim->tris.attr<vec3f>("uv2");
        auto &loop_uv = prim->loops.add_attr<vec3f>("uv");
        for (int i = 0; i < prim->tris.size(); i++) {
            loop_uv[old_loop_base + i * 3 + 0] = uv0[i];
            loop_uv[old_loop_base + i * 3 + 1] = uv1[i];
            loop_uv[old_loop_base + i * 3 + 2] = uv2[i];
        }
    }

    prim->tris.clear();
    prim->quads.clear();
}

namespace {

struct PrimitivePolygonate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primPolygonate(prim.get(), get_param<bool>("with_uv"));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitivePolygonate,
        { /* inputs: */ {
        {"primitive", "prim"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        {"bool", "with_uv", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
