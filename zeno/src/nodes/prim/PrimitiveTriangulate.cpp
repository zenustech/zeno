#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>

namespace zeno {

ZENO_API void primTriangulateQuads(PrimitiveObject *prim) {
    auto base = prim->tris.size();
    prim->tris.resize(base + prim->quads.size() * 2);

    for (size_t i = 0; i < prim->quads.size(); i++) {
        auto quad = prim->quads[i];
        prim->tris[base+i*2+0] = vec3f(quad[0], quad[1], quad[2]);
        prim->tris[base+i*2+1] = vec3f(quad[0], quad[2], quad[3]);
    }
    prim->quads.clear();
}

ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv) {
    prim->tris.reserve(prim->tris.size() + prim->polys.size());

    std::vector<int> scansum(prim->polys.size());
    auto redsum = parallel_exclusive_scan_sum(prim->polys.begin(), prim->polys.end(),
                                           scansum.begin(), [&] (auto &ind) {
                                               return ind[1] >= 3 ? ind[1] : 0;
                                           });
    auto tribase = prim->tris.size();
    prim->tris.resize(tribase + redsum);

    if (!prim->loops.has_attr("uv") || !with_uv) {
        parallel_for(prim->polys.size(), [&] (size_t i) {
            auto [start, len] = prim->polys[i];
            int scanbase = scansum[i] + tribase;
            prim->tris[scanbase++] = vec3f(
                    prim->loops[start],
                    prim->loops[start + 1],
                    prim->loops[start + 2]);
            for (int j = 3; j < len; j++) {
                prim->tris[scanbase++] = vec3f(
                        prim->loops[start],
                        prim->loops[start + j - 1],
                        prim->loops[start + j]);
            }
        });

    } else {
        auto &loop_uv = prim->loops.attr<vec3f>("uv");
        auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<vec3f>("uv2");

        parallel_for(prim->polys.size(), [&] (size_t i) {
            auto [start, len] = prim->polys[i];
            int scanbase = scansum[i] + tribase;
            uv0[scanbase] = loop_uv[start];
            uv1[scanbase] = loop_uv[start + 1];
            uv2[scanbase] = loop_uv[start + 2];
            prim->tris[scanbase++] = vec3f(
                    prim->loops[start],
                    prim->loops[start + 1],
                    prim->loops[start + 2]);
            for (int j = 3; j < len; j++) {
                uv0[scanbase] = loop_uv[start];
                uv1[scanbase] = loop_uv[start + j - 1];
                uv2[scanbase] = loop_uv[start + j];
                prim->tris[scanbase++] = vec3f(
                        prim->loops[start],
                        prim->loops[start + j - 1],
                        prim->loops[start + j]);
            }
        });

    }
    prim->loops.clear();
    prim->polys.clear();
}

namespace {

struct PrimitiveTriangulate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (get_param<bool>("from_poly")) {
            primTriangulate(prim.get(), get_param<bool>("with_uv"));
        }
        if (get_param<bool>("from_quads")) {
            primTriangulateQuads(prim.get());
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
