#include "zensim/container/Bht.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include <cstdio>
#include <cstring>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/zeno.h>

namespace zeno {

struct TrianglePrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int niters = get_input2<int>("iters");
        auto faceInherentTag = get_input2<std::string>("face_inherit");
        auto faceAvgEdgeTag = get_input2<std::string>("face_avg_edge");

        std::set<std::string> faceInherentPropTags, faceAvgEdgePropTags;
        auto processTags = [](std::string tags, std::set<std::string> &res) {
            using Ti = RM_CVREF_T(std::string::npos);
            Ti st = tags.find_first_not_of(", ", 0);
            for (auto ed = tags.find_first_of(", ", st + 1); ed != std::string::npos;
                 ed = tags.find_first_of(", ", st + 1)) {
                res.insert(tags.substr(st, ed - st));
                // fmt::print("extract [{}, {}): [{}]\n", st, ed, res.back());
                st = tags.find_first_not_of(", ", ed);
                if (st == std::string::npos)
                    break;
            }
            if (st != std::string::npos && st < tags.size()) {
                res.insert(tags.substr(st));
                // fmt::print("extract [{}, npos): [{}]\n", st, res.back());
            }
        };
        processTags(faceInherentTag, faceInherentPropTags);
        fmt::print("faceInherentTag: ");
        for (auto str : faceInherentPropTags)
            fmt::print("[{}]\t", str);
        processTags(faceAvgEdgeTag, faceAvgEdgePropTags);
        fmt::print("\nfaceInherentTag: ");
        for (auto str : faceAvgEdgePropTags)
            fmt::print("[{}]\t", str);
        fmt::print("\n");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &verts = prim->verts;
        auto &tris = prim->tris;
        bool handleTriUV = tris.has_attr("uv0") && tris.has_attr("uv1") && tris.has_attr("uv2");
        using vec2i = zs::vec<int, 2>;
        for (int i = 0; i != niters; ++i) {
            bht<int, 2, int> tab{3 * tris.size() * 2};
            tab.reset(pol, true);
            pol(tris.values, [tab = proxy<space>(tab)](auto tri) mutable {
                int u = tri[2];
                for (int d = 0; d != 3; ++d) {
                    int v = tri[d];
                    tab.insert(vec2i{std::min(u, v), std::max(u, v)});
                    u = v;
                }
            });

            /// @note verts
            int vOffset = verts.size();
            auto numNewVerts = tab.size();
            verts.resize(vOffset + numNewVerts);
            pol(range(numNewVerts), [&, edges = proxy<space>(tab._activeKeys), vOffset](int ei) mutable {
                auto edge = edges[ei];
                auto p = (verts.values[edge[0]] + verts.values[edge[1]]) / 2;
                verts.values[vOffset + ei] = p;
                verts.foreach_attr<AttrAcceptAll>(
                    [&](auto const &key, auto &arr) { arr[vOffset + ei] = (arr[edge[0]] + arr[edge[1]]) / 2; });
            });

            /// @note tris
            int tOffset = tris.size();
            auto numNewTris = tris.size() * 3; // each one divided into 4 pieces
            tris.resize(tOffset + numNewTris);
            auto assignSubTriAttr = [&tris, tOffset](auto tag_c, std::string tag, int ti) {
                using TT = typename RM_CVREF_T(tag_c)::type;
                auto &tag0 = tris.attr<TT>(tag + "0");
                auto &tag1 = tris.attr<TT>(tag + "1");
                auto &tag2 = tris.attr<TT>(tag + "2");
                TT props[3] = {tag0[ti], tag1[ti], tag2[ti]};
                TT mid_props[3] = {(props[2] + props[0]) / 2, (props[0] + props[1]) / 2, (props[1] + props[2]) / 2};
                tag0[ti] = mid_props[0];
                tag1[ti] = mid_props[1];
                tag2[ti] = mid_props[2];
                tag0[tOffset + ti * 3 + 0] = props[0];
                tag1[tOffset + ti * 3 + 0] = mid_props[1];
                tag2[tOffset + ti * 3 + 0] = mid_props[0];
                tag0[tOffset + ti * 3 + 1] = props[1];
                tag1[tOffset + ti * 3 + 1] = mid_props[2];
                tag2[tOffset + ti * 3 + 1] = mid_props[1];
                tag0[tOffset + ti * 3 + 2] = props[2];
                tag1[tOffset + ti * 3 + 2] = mid_props[0];
                tag2[tOffset + ti * 3 + 2] = mid_props[2];
            };
            pol(range(tOffset), [&, tab = proxy<space>(tab), vOffset, tOffset](int ti) mutable {
                ///
                auto tri = tris.values[ti];
                int u = tri[2];
                int ids[3]; // tri[2] - {ids[0]} - tri[0] - {ids[1]} - tri[1] - {ids[2]} - tri[2]
                for (int d = 0; d != 3; ++d) {
                    int v = tri[d];
                    auto ei = tab.query(vec2i{std::min(u, v), std::max(u, v)});
                    ids[d] = vOffset + ei;
                    u = v;
                }
                tris.values[ti] = zeno::vec3i{ids[0], ids[1], ids[2]};
                tris.values[tOffset + ti * 3 + 0] = zeno::vec3i{tri[0], ids[1], ids[0]};
                tris.values[tOffset + ti * 3 + 1] = zeno::vec3i{tri[1], ids[2], ids[1]};
                tris.values[tOffset + ti * 3 + 2] = zeno::vec3i{tri[2], ids[0], ids[2]};

                if (handleTriUV) {
                    assignSubTriAttr(wrapt<zeno::vec3f>{}, "uv", ti);
                }
                tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                    using TT = RM_CVREF_T(arr[0]);
                    if (key == "uv0" || key == "uv1" || key == "uv2")
                        return;
                    // avg_by_edge
                    if (auto k = key.substr(0, key.size() - 1);
                        faceAvgEdgePropTags.find(k) != faceAvgEdgePropTags.end() &&
                        (key.back() == '0' || key.back() == '1' || key.back() == '2') && k != "uv") {
                        assignSubTriAttr(zs::wrapt<TT>{}, k, ti);
                        return;
                    }
                    auto val = arr[ti]; // inherent
                    if (faceInherentPropTags.find(key) == faceInherentPropTags.end())
                        val /= 4; // average
                    arr[ti] = val;
                    arr[tOffset + ti * 3 + 0] = val;
                    arr[tOffset + ti * 3 + 1] = val;
                    arr[tOffset + ti * 3 + 2] = val;
                });
            });
        }
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(TrianglePrimSubdiv)
({
    {
        "prim",
        {"int", "iters", "2"},
        {"string", "face_inherit", ""},
        {"string", "face_avg_edge", ""},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

} // namespace zeno