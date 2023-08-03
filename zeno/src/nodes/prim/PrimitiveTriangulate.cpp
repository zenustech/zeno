#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/parallel_scan.h>
#include <zeno/utils/variantswitch.h>

namespace zeno {

ZENO_API void primTriangulateQuads(PrimitiveObject *prim) {
    auto base = prim->tris.size();
    prim->tris.resize(base + prim->quads.size() * 2);
    bool hasmat = prim->quads.has_attr("matid");
    if(hasmat == false)
    {
        prim->quads.add_attr<int>("matid");
        prim->quads.attr<int>("matid").assign(prim->quads.size(), -1);
    }

    if (prim->tris.has_attr("matid")) {
        prim->tris.attr<int>("matid").resize(base + prim->quads.size() * 2);
    } else {
        prim->tris.add_attr<int>("matid");
    }


    for (size_t i = 0; i < prim->quads.size(); i++) {
        auto quad = prim->quads[i];
        prim->tris[base+i*2+0] = vec3f(quad[0], quad[1], quad[2]);
        prim->tris[base+i*2+1] = vec3f(quad[0], quad[2], quad[3]);
        if(hasmat) {
            prim->tris.attr<int>("matid")[base + i * 2 + 0] = prim->quads.attr<int>("matid")[i];
            prim->tris.attr<int>("matid")[base + i * 2 + 1] = prim->quads.attr<int>("matid")[i];
        } else
        {
            prim->tris.attr<int>("matid")[base + i * 2 + 0] = -1;
            prim->tris.attr<int>("matid")[base + i * 2 + 1] = -1;
        }
    }
    prim->quads.clear();
}

ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv, bool has_lines) {
    //prim->tris.reserve(prim->tris.size() + prim->polys.size());
    bool hasmat = prim->polys.has_attr("matid");
    if(!hasmat)
    {
        prim->polys.add_attr<int>("matid");
        prim->polys.attr<int>("matid").assign(prim->polys.size(), -1);
    }
  boolean_switch(has_lines, [&] (auto has_lines) {
    std::vector<std::conditional_t<has_lines.value, vec2i, int>> scansum(prim->polys.size());
    auto redsum = parallel_exclusive_scan_sum(prim->polys.begin(), prim->polys.end(),
                                           scansum.begin(), [&] (auto &ind) {
                                               if constexpr (has_lines.value) {
                                                   return vec2i(ind[1] >= 3 ? ind[1] - 2 : 0, ind[1] == 2 ? 1 : 0);
                                               } else {
                                                   return ind[1] >= 3 ? ind[1] - 2 : 0;
                                               }
                                           });
    int tribase = prim->tris.size();
    int linebase = prim->lines.size();
    if constexpr (has_lines.value) {
        prim->tris.resize(tribase + redsum[0]);
        prim->lines.resize(linebase + redsum[1]);
    } else {
        prim->tris.resize(tribase + redsum);
    }

    if (prim->tris.has_attr("matid")) {
        prim->tris.attr<int>("matid").resize(prim->tris.size());
    } else {
        prim->tris.add_attr<int>("matid");
    }


    if (!(prim->loops.has_attr("uvs") && prim->uvs.size() > 0) || !with_uv) {
        parallel_for(prim->polys.size(), [&] (size_t i) {
            auto [start, len] = prim->polys[i];
            auto matidx = prim->polys.attr<int>("matid")[i];
            if (len >= 3) {
                int scanbase;
                if constexpr (has_lines.value) {
                    scanbase = scansum[i][0] + tribase;
                } else {
                    scanbase = scansum[i] + tribase;
                }
                prim->tris[scanbase] = vec3i(
                        prim->loops[start],
                        prim->loops[start + 1],
                        prim->loops[start + 2]);
                prim->tris.attr<int>("matid")[scanbase] = matidx;
                scanbase++;
                for (int j = 3; j < len; j++) {
                    prim->tris[scanbase] = vec3i(
                            prim->loops[start],
                            prim->loops[start + j - 1],
                            prim->loops[start + j]);
                    prim->tris.attr<int>("matid")[scanbase] = matidx;
                    scanbase++;
                }
            }
            if constexpr (has_lines.value) {
                if (len == 2) {
                    int scanbase = scansum[i][1] + linebase;
                    prim->lines[scanbase] = vec2i(
                        prim->loops[start],
                        prim->loops[start + 1]);
                }
            }
        });

    } else {
        auto &loop_uv = prim->loops.attr<int>("uvs");
        auto &uvs = prim->uvs;
        auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<vec3f>("uv2");

        parallel_for(prim->polys.size(), [&] (size_t i) {
            auto [start, len] = prim->polys[i];
            auto matidx = prim->polys.attr<int>("matid")[i];
            if (len >= 3) {
                int scanbase;
                if constexpr (has_lines.value) {
                    scanbase = scansum[i][0] + tribase;
                } else {
                    scanbase = scansum[i] + tribase;
                }
                uv0[scanbase] = {uvs[loop_uv[start]][0], uvs[loop_uv[start]][1], 0};
                uv1[scanbase] = {uvs[loop_uv[start + 1]][0], uvs[loop_uv[start + 1]][1], 0};
                uv2[scanbase] = {uvs[loop_uv[start + 2]][0], uvs[loop_uv[start + 2]][1], 0};
                prim->tris[scanbase] = vec3i(
                        prim->loops[start],
                        prim->loops[start + 1],
                        prim->loops[start + 2]);
                prim->tris.attr<int>("matid")[scanbase] = matidx;
                scanbase++;
                for (int j = 3; j < len; j++) {
                    uv0[scanbase] = {uvs[loop_uv[start]][0], uvs[loop_uv[start]][1], 0};
                    uv1[scanbase] = {uvs[loop_uv[start + j - 1]][0], uvs[loop_uv[start + j - 1]][1], 0};
                    uv2[scanbase] = {uvs[loop_uv[start + j]][0], uvs[loop_uv[start + j]][1], 0};
                    prim->tris[scanbase] = vec3i(
                            prim->loops[start],
                            prim->loops[start + j - 1],
                            prim->loops[start + j]);
                    prim->tris.attr<int>("matid")[scanbase] = matidx;
                    scanbase++;
                }
            }
            if constexpr (has_lines.value) {
                if (len == 2) {
                    int scanbase = scansum[i][1] + linebase;
                    prim->lines[scanbase] = vec2i(
                        prim->loops[start],
                        prim->loops[start + 1]);
                }
            }
        });

    }
    prim->loops.clear();
    prim->polys.clear();
    prim->loops.erase_attr("uvs");
    prim->uvs.clear();
  });
}

namespace {

struct PrimitiveTriangulate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        if (get_param<bool>("from_poly")) {
            primTriangulate(prim.get(), get_param<bool>("with_uv"), get_param<bool>("has_lines"));
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
        {"bool", "has_lines", "1"},
        }, /* category: */ {
        "primitive",
        }});

}
}
