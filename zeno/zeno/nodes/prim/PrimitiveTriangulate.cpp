#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>

namespace zeno {

ZENO_API void prim_quads_to_tris(PrimitiveObject *prim) {
    prim->tris.reserve(prim->tris.size() + prim->quads.size() * 2);

    for (auto quad: prim->quads) {
        prim->tris.emplace_back(quad[0], quad[1], quad[2]);
        prim->tris.emplace_back(quad[0], quad[2], quad[3]);
    }
    prim->quads.clear();
}

ZENO_API void prim_polys_to_tris(PrimitiveObject *prim) {
    prim->tris.reserve(prim->tris.size() + prim->polys.size());

    for (auto [start, len]: prim->polys) {
        if (len >= 3) {
            prim->tris.emplace_back(
                    prim->loops[start],
                    prim->loops[start + 1],
                    prim->loops[start + 2]);
            for (int i = 3; i < len; i++) {
                prim->tris.emplace_back(
                        prim->loops[start],
                        prim->loops[start + i - 1],
                        prim->loops[start + i]);
            }
        } else if (len == 2) {
            prim->lines.emplace_back(
                    prim->loops[start],
                    prim->loops[start + 1]);
        }
    }
    prim->loops.clear();
}

ZENO_API void prim_polys_to_tris_with_uv(PrimitiveObject *prim) {
    if (!prim->loops.has_attr("uv"))
        return prim_polys_to_tris(prim);

    auto &loop_uv = prim->loops.attr<vec3f>("uv");

    auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
    auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
    auto &uv2 = prim->tris.add_attr<vec3f>("uv2");

    auto &line_uv0 = prim->lines.add_attr<vec3f>("uv0");
    auto &line_uv1 = prim->lines.add_attr<vec3f>("uv1");

    prim->tris.reserve(prim->tris.size() + prim->polys.size());
    uv0.reserve(uv0.size() + prim->polys.size());
    uv1.reserve(uv1.size() + prim->polys.size());
    uv2.reserve(uv2.size() + prim->polys.size());

    for (auto [start, len]: prim->polys) {
        if (len >= 3) {
            uv0.push_back(loop_uv[start]);
            uv1.push_back(loop_uv[start + 1]);
            uv2.push_back(loop_uv[start + 2]);
            prim->tris.emplace_back(
                    prim->loops[start],
                    prim->loops[start + 1],
                    prim->loops[start + 2]);
            for (int i = 3; i < len; i++) {
                uv0.push_back(loop_uv[start]);
                uv1.push_back(loop_uv[start + i - 1]);
                uv2.push_back(loop_uv[start + i]);
                prim->tris.emplace_back(
                        prim->loops[start],
                        prim->loops[start + i - 1],
                        prim->loops[start + i]);
            }
        } else if (len == 2) {
            line_uv0.push_back(loop_uv[start]);
            line_uv1.push_back(loop_uv[start + 1]);
            prim->lines.emplace_back(
                    prim->loops[start],
                    prim->loops[start + 1]);
        }
    }
    prim->loops.clear();
}

namespace {

struct PrimitiveTriangulate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (get_param<bool>("from_poly")) {
            if (get_param<bool>("with_uv")) {
                prim_polys_to_tris_with_uv(prim.get());
            } else {
                prim_polys_to_tris(prim.get());
            }
        }
        if (get_param<bool>("from_quads")) {
            prim_quads_to_tris(prim.get());
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimitiveTriangulate,
        { /* inputs: */ {
        {"primitive", "prim"},
        }, /* outputs: */ {
        {"primitive", "prim"},
        }, /* params: */ {
        {"bool", "from_poly", "1"},
        {"bool", "with_uv", "1"},
        {"bool", "from_quads", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
