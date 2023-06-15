#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>

#include "zensim/container/Bht.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"

namespace zeno {

struct PolyReduceLite : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");

        using namespace zs;
        constexpr auto space = execspace_e::openmp;
        auto pol = omp_exec();

        auto &verts = prim->verts;
        auto &pos = verts.values;
        std::vector<int> vertDiscard(pos.size());
        std::vector<std::set<int>> vertTris(pos.size());  /// neighboring tris
        std::vector<std::set<int>> vertVerts(pos.size()); /// neighboring verts

        auto &tris = prim->tris.values;
        std::vector<int> triDiscard(tris.size());

        {
            /// establish vert-face & vert-vert relations
            std::vector<std::mutex> vertMutex(pos.size());
            pol(enumerate(tris), [&](int triNo, auto tri) {
                auto vNo = tri[2];
                for (int d = 0; d != 3; ++d) {
                    auto vNoJ = tri[d];
                    {
                        std::lock_guard lk(vertMutex[vNo]);
                        vertTris[vNo].insert(triNo);
                        vertVerts[vNo].insert(vNoJ);
                    }
                    {
                        std::lock_guard lk(vertMutex[vNoJ]);
                        vertVerts[vNoJ].insert(vNo);
                    }
                    vNo = vNoJ;
                }
            });
        }

        int nIters = get_input2<int>("iterations");

        auto triHasVert = [&tris](const auto &tri, int v) {
            for (auto vNo : tri)
                if (vNo == v)
                    return true;
            return false;
        };
        int u, v;
        std::mt19937 rng;
        rng.seed(0);
        for (int i = 0; i != nIters; ++i) {
            zeno::log_warn(fmt::format("begin iter {}\n", i));
            /// evaluate vert curvatures
            ;
            /// sort edges for collapse
            ;
            // temporal measure
            do {
                u = (u32)rng() % (u32)pos.size();
                v = -1;
                for (auto nv : vertVerts[u])
                    if (!vertDiscard[nv]) {
                        v = nv;
                    }
            } while (vertDiscard[u] && v == -1);
            if (u < v) {
                std::swap(u, v);
            }
            fmt::print("collapsing {} to {}.\n", u, v);

            // 0. adjust v pos
            pos[v] = (pos[v] + pos[u]) / 2;
            // 1. remove vert u (also maintain vertVerts)
            vertDiscard[u] = 1;
            for (auto nv : vertVerts[u]) {
                vertVerts[nv].erase(u);
                if (nv != v) {
                    vertVerts[nv].insert(v);
                    vertVerts[v].insert(nv);
                }
            }
            zeno::log_warn(fmt::format("done phase 1\n"));
            // 2. remove triangles containing edge <u, v>
            for (int triNo : vertTris[u]) {
                if (triDiscard[triNo])
                    continue;
                if (triHasVert(tris[triNo], v)) {
                    // delete this triangle
                    triDiscard[triNo] = 1;
                }
            }
            zeno::log_warn(fmt::format("done phase 2\n"));
            // 3. remapping triangles verts u->v (also maintain vertTris)
            for (auto triNo : vertTris[u]) {
                if (triDiscard[triNo])
                    continue;
                auto &tri = tris[triNo];
                for (auto &id : tri)
                    if (id == u) {
                        id = v;
                        vertTris[v].insert(triNo);
                        break;
                    }
            }
            zeno::log_warn(fmt::format("done phase 3\n"));
        }

        std::vector<int> voffsets(pos.size());
        {
            auto &vertPreserve = vertDiscard;
            /// compact verts
            pol(vertPreserve, [](int &v) { v = !v; });
            exclusive_scan(pol, std::begin(vertPreserve), std::end(vertPreserve), std::begin(voffsets));
            auto nvs = voffsets.back() + vertPreserve.back();
            fmt::print("{} verts to {} verts\n", pos.size(), nvs);
            RM_CVREF_T(prim->verts) newVerts;
            newVerts.resize(nvs);
            pol(range(pos.size()), [&](int i) {
                if (vertPreserve[i])
                    newVerts.values[voffsets[i]] = pos[i];
            });
            prim->verts = std::move(newVerts);
        }
        {
            std::vector<int> offsets(tris.size());
            auto &triPreserve = triDiscard;
            /// compact tris and map vert indices
            pol(triPreserve, [](int &v) { v = !v; });
            exclusive_scan(pol, std::begin(triPreserve), std::end(triPreserve), std::begin(offsets));
            auto nts = offsets.back() + triPreserve.back();
            fmt::print("{} tris to {} tris\n", tris.size(), nts);
            RM_CVREF_T(prim->tris) newTris;
            newTris.resize(nts);
            pol(range(tris.size()), [&](int i) {
                if (triPreserve[i]) {
                    auto tri = tris[i];
                    for (auto &v : tri)
                        v = voffsets[v];
                    fmt::print("preserving tri [{}] <{}, {}, {}>\n", i, tri[0], tri[1], tri[2]);
                    newTris.values[offsets[i]] = tri;
                }
            });
            prim->tris = std::move(newTris);
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PolyReduceLite, {
                               {{"PrimitiveObject", "prim"}, {"int", "iterations", "100"}},
                               {
                                   {"PrimitiveObject", "prim"},
                               },
                               {},
                               {"zs_geom"},
                           });

} // namespace zeno