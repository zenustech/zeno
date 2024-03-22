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

ZENO_API void primTriangulate(PrimitiveObject *prim, bool with_uv, bool has_lines, bool with_attr) {
    if (prim->polys.size() == 0) {
        return;
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
    std::vector<int> mapping;
    int tribase = prim->tris.size();
    int linebase = prim->lines.size();
    if constexpr (has_lines.value) {
        prim->tris.resize(tribase + redsum[0]);
        mapping.resize(tribase + redsum[0]);
        prim->lines.resize(linebase + redsum[1]);
    } else {
        prim->tris.resize(tribase + redsum);
        mapping.resize(tribase + redsum);
    }

    if (!(prim->loops.has_attr("uvs") && prim->uvs.size() > 0) || !with_uv) {
        parallel_for(prim->polys.size(), [&] (size_t i) {
            auto [start, len] = prim->polys[i];

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
                mapping[scanbase] = i;
                scanbase++;
                for (int j = 3; j < len; j++) {
                    prim->tris[scanbase] = vec3i(
                            prim->loops[start],
                            prim->loops[start + j - 1],
                            prim->loops[start + j]);
                    mapping[scanbase] = i;
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
                mapping[scanbase] = i;
                scanbase++;
                for (int j = 3; j < len; j++) {
                    uv0[scanbase] = {uvs[loop_uv[start]][0], uvs[loop_uv[start]][1], 0};
                    uv1[scanbase] = {uvs[loop_uv[start + j - 1]][0], uvs[loop_uv[start + j - 1]][1], 0};
                    uv2[scanbase] = {uvs[loop_uv[start + j]][0], uvs[loop_uv[start + j]][1], 0};
                    prim->tris[scanbase] = vec3i(
                            prim->loops[start],
                            prim->loops[start + j - 1],
                            prim->loops[start + j]);
                    mapping[scanbase] = i;
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
    if (with_attr) {
        prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
          using T = std::decay_t<decltype(arr[0])>;
          auto &attr = prim->tris.add_attr<T>(key);
          for (auto i = tribase; i < attr.size(); i++) {
              attr[i] = arr[mapping[i]];
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
            primTriangulate(prim.get(), get_param<bool>("with_uv"), get_param<bool>("has_lines"), get_input2<bool>("with_attr"));
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
        {"bool", "with_attr", "1"},
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

ZENO_API void primTriangulateIntoPolys(PrimitiveObject *prim) {
    if (prim->tris.size()) {
        primPolygonate(prim, true);
    }
    else if (prim->polys.size()) {
        int new_poly_count = 0;
        int new_loops_count = 0;
        for (auto [_s, c] : prim->polys) {
            new_poly_count += c > 3 ? c - 2 : 1;
            new_loops_count += c > 3 ? 3 * (c - 2) : c;
        }

        {
            AttrVector<int> loops;
            loops.values.reserve(new_loops_count);
            std::vector<int> loops_mapping_old;
            loops_mapping_old.reserve(new_loops_count);
            for (auto j = 0; j < prim->polys.size(); j++) {
                auto [s, c] = prim->polys[j];
                if (c > 3) {
                    for (auto i = 0; i < c - 2; i++) {
                        loops.emplace_back(prim->loops[s]);
                        loops.emplace_back(prim->loops[s + 1 + i]);
                        loops.emplace_back(prim->loops[s + 2 + i]);
                        loops_mapping_old.emplace_back(s);
                        loops_mapping_old.emplace_back(s + 1 + i);
                        loops_mapping_old.emplace_back(s + 2 + i);
                    }
                }
                else {
                    for (auto i = s; i < s + c; i++) {
                        loops.emplace_back(prim->loops[i]);
                        loops_mapping_old.emplace_back(i);
                    }
                }
            }
            prim->loops.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = loops.add_attr<T>(key);
                for (auto i = 0; i < attr.size(); i++) {
                    attr[i] = arr[loops_mapping_old[i]];
                }
            });
            prim->loops = loops;
        }
        {
            AttrVector<vec2i> polys;
            polys.values.reserve(new_poly_count);
            std::vector<int> polys_mapping_old;
            polys_mapping_old.reserve(new_poly_count);
            for (auto j = 0; j < prim->polys.size(); j++) {
                auto [_s, c] = prim->polys[j];
                if (c > 3) {
                    for (auto i = 0; i < c - 2; i++) {
                        polys.values.emplace_back(0, 3);
                        polys_mapping_old.emplace_back(j);
                    }
                }
                else {
                    polys.values.emplace_back(0, c);
                    polys_mapping_old.emplace_back(j);
                }
            }
            int start = 0;
            for (auto i = 0; i < polys.size(); i++) {
                auto [_s, c] = polys[i];
                polys[i] = {start, c};
                start += c;
            }
            prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = polys.add_attr<T>(key);
                for (auto i = 0; i < attr.size(); i++) {
                    attr[i] = arr[polys_mapping_old[i]];
                }
            });
            prim->polys = polys;
        }
    }
}

struct PrimTriangulateIntoPolys : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        primTriangulateIntoPolys(prim.get());
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimTriangulateIntoPolys,
        { /* inputs: */ {
            {"primitive", "prim"},
        }, /* outputs: */ {
            {"primitive", "prim"},
        }, /* params: */ {
        }, /* category: */ {
            "primitive",
        }});
}
