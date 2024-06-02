#include <zeno/core/Graph.h>
#include <zeno/extra/assetDir.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/log.h>
#include <zeno/zeno.h>
#include <mutex>

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
        std::vector<std::set<int>> vertTris(pos.size());                              /// neighboring tris
        std::vector<std::set<int>> vertVerts(pos.size());                             /// neighboring verts
        std::vector<std::pair<float, std::pair<int, int>>> vertEdgeCosts(pos.size()); /// neighboring verts

        auto &tris = prim->tris.values;
        std::vector<int> triDiscard(tris.size());
        std::vector<zeno::vec3f> triNorms(tris.size());

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
                triNorms[triNo] = normalize(cross(pos[tri[1]] - pos[tri[0]], pos[tri[2]] - pos[tri[0]]));
            });
        }

        int nIters = get_input2<int>("iterations");

        auto updateTriNormal = [&tris, &pos, &triNorms](int triNo, const auto &tri) {
            triNorms[triNo] = normalize(cross(pos[tri[1]] - pos[tri[0]], pos[tri[2]] - pos[tri[0]]));
        };
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
            // zeno::log_warn(fmt::format("begin iter {}\n", i));
            /// evaluate vert curvatures
            pol(range(pos.size()), [&](int i) {
                vertEdgeCosts[i] = std::make_pair(limits<float>::max(), std::make_pair(i, -1));
                if (vertVerts[i].size() == 0 || vertDiscard[i]) {
                    return;
                }

                auto cost = limits<float>::max();
                for (auto j : vertVerts[i]) {
                    if (vertDiscard[j])
                        continue;
                    auto elen = length(pos[i] - pos[j]);
                    auto curvature = 0.f;
                    std::vector<int> sides; // tris that owns edge <i, j>
                    for (int triNo : vertTris[i]) {
                        if (triDiscard[triNo])
                            continue;
                        if (triHasVert(tris[triNo], j)) {
                            sides.push_back(triNo);
                        }
                    }

                    for (int triI : vertTris[i]) {
                        auto minCurv = 1.f;
                        for (auto triJ : sides) {
                            auto dotProd = dot(triNorms[triI], triNorms[triJ]);
                            minCurv = std::min(minCurv, (1 - dotProd) / 2.f);
                        }
                        curvature = std::max(curvature, minCurv);
                    }
                    if (auto c = curvature * elen; c < cost) {
                        vertEdgeCosts[i] = std::make_pair(c, std::make_pair(i, j));
                    }
                }
            });
            /// sort edges for collapse
            auto pair = std::reduce(
                std::begin(vertEdgeCosts), std::end(vertEdgeCosts),
                std::make_pair(limits<float>::max(), std::make_pair(-1, -1)),
                [](const std::pair<float, std::pair<int, int>> &a, const std::pair<float, std::pair<int, int>> &b) {
                    if (a.first < b.first)
                        return a;
                    else
                        return b;
                });
#if 1
            u = pair.second.first;
            v = pair.second.second;
            // fmt::print("selecting uv <{}, {}>\n", u, v);
            if (v == -1) {
                fmt::print("no more edges to collapse!\n");
                break;
            }
            // pos[v] = (pos[v] + pos[u]) / 2;
#else
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
            // 0. adjust v pos
            pos[v] = (pos[v] + pos[u]) / 2;
#endif
            // fmt::print("collapsing {} to {}.\n", u, v);

            // 1. remove vert u (also maintain vertVerts)
            vertDiscard[u] = 1;
            for (auto nv : vertVerts[u]) {
                vertVerts[nv].erase(u);
                if (nv != v) {
                    vertVerts[nv].insert(v);
                    vertVerts[v].insert(nv);
                }
            }
            // zeno::log_warn(fmt::format("done phase 1\n"));
            // 2. remove triangles containing edge <u, v>
            for (int triNo : vertTris[u]) {
                if (triDiscard[triNo])
                    continue;
                if (triHasVert(tris[triNo], v)) {
                    // delete this triangle
                    triDiscard[triNo] = 1;
                }
            }
            // zeno::log_warn(fmt::format("done phase 2\n"));
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
            // zeno::log_warn(fmt::format("done phase 3\n"));
            // 4. update tri normals
            for (auto triNo : vertTris[v]) {
                if (triDiscard[triNo])
                    continue;
                auto &tri = tris[triNo];
                updateTriNormal(triNo, tri);
            }
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