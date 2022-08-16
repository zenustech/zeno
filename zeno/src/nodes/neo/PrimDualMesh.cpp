#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/utils/log.h>
#include <cmath>

namespace zeno {
namespace {

template <class T>
struct meth_average {
    T value{0};
    int count{0};

    void add(T x) {
        value += x;
        ++count;
    }

    T get() {
        return value / (T)count;
    }
};

/*template <class Func>
static void prim_foreach_faces_corns(PrimitiveObject *prim, Func const &each_face) {
    for (int i = 0; i < prim->tris.size(); i++) {
        each_face(i, [&] (auto const &each_corn) {
            auto [f1, f2, f3] = prim->tris[i];
            each_corn(f1);
            each_corn(f2);
            each_corn(f3);
        });
    }
    for (int i = 0; i < prim->quads.size(); i++) {
        each_face(i + prim->tris.size(), [&] (auto const &each_corn) {
            auto [f1, f2, f3, f4] = prim->quads[i];
            each_corn(f1);
            each_corn(f2);
            each_corn(f3);
            each_corn(f4);
        });
    }
    for (int i = 0; i < prim->polys.size(); i++) {
        each_face(i + prim->tris.size() + prim->quads.size(), [&] (auto const &each_corn) {
            auto [start, len] = prim->polys[i];
            for (int i = start; i < start + len; i++) {
                each_corn(prim->loops[i]);
            }
        });
    }
}

template <class Func>
static void prim_foreach_faces_edges(PrimitiveObject *prim, Func const &each_face) {
    for (int i = 0; i < prim->tris.size(); i++) {
        each_face(i, [&] (auto const &each_edge) {
            auto [f1, f2, f3] = prim->tris[i];
            each_edge(f1, f2);
            each_edge(f2, f3);
            each_edge(f3, f1);
        });
    }
    for (int i = 0; i < prim->quads.size(); i++) {
        each_face(i + prim->tris.size(), [&] (auto const &each_edge) {
            auto [f1, f2, f3, f4] = prim->quads[i];
            each_edge(f1, f2);
            each_edge(f2, f3);
            each_edge(f3, f4);
            each_edge(f4, f1);
        });
    }
    for (int i = 0; i < prim->polys.size(); i++) {
        each_face(i + prim->tris.size() + prim->quads.size(), [&] (auto const &each_edge) {
            auto [start, len] = prim->polys[i];
            if (len >= 1) {
                for (int i = start; i < start + len - 1; i++) {
                    each_edge(prim->loops[i], prim->loops[i + 1]);
                }
                each_edge(prim->loops[start + len - 1], prim->loops[0]);
            }
        });
    }
}*/

struct PrimDualMesh : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        //auto faceType = get_input2<std::string>("faceType");
        //auto copyFaceAttrs = get_input2<bool>("copyFaceAttrs");
        auto outprim = std::make_shared<PrimitiveObject>();
        bool keepBounds = get_input2<bool>("keepBounds");

        if (get_input2<bool>("polygonate")/* && (prim->tris.size() || prim->quads.size())*/) {
            //prim = std::make_shared<PrimitiveObject>(*prim);
            //prim->lines.clear();
            primPolygonate(prim.get());
        }

        scope_exit<> revertoldpolysize;
        if (keepBounds) {
            std::map<std::pair<int, int>, int> bounde2f;
            {
                std::vector<std::pair<int, int>> notbound;
                for (int f = 0; f < prim->polys.size(); f++) {
                    auto each_edge = [&] (int v1, int v2) {
                        std::pair<int, int> key(std::min(v1, v2), std::max(v1, v2));
                        auto [it, succ] = bounde2f.try_emplace(key, f);
                        if (!succ) {
                            notbound.push_back(key);
                            it->second = -1;
                        }
                    };
                    auto [start, len] = prim->polys[f];
                    if (len >= 1) {
                        for (int l = start; l < start + len - 1; l++) {
                            each_edge(prim->loops[l], prim->loops[l + 1]);
                        }
                        each_edge(prim->loops[start + len - 1], prim->loops[start]);
                    }
                }
                for (auto const &nbk: notbound)
                    bounde2f.erase(nbk);
            }
            auto oldpolysize = prim->polys.size();
            revertoldpolysize = scope_exit<>([prim, oldpolysize] {
                prim->polys.resize(oldpolysize);
            });
            for (auto const &[ev, f]: bounde2f) {
                auto [v1, v2] = ev;
                int loopbase = prim->loops.size();
                prim->loops.push_back(v1);
                prim->loops.push_back(v2);
                prim->polys.emplace_back(loopbase, 2);
            }
        }

        std::map<int, std::vector<int>> v2f;
        outprim->verts.resize(prim->polys.size());
        for (int f = 0; f < prim->polys.size(); f++) {
            meth_average<vec3f> reducer;
            auto [start, len] = prim->polys[f];
            for (int l = start; l < start + len; l++) {
                int v = prim->loops[l];
                reducer.add(prim->verts[v]);
                v2f[v].push_back(f);
            }
            outprim->verts[f] = reducer.get();
        }

        std::for_each(v2f.begin(), v2f.end(), [&] (auto const &v2fent) {
            auto const &[vid, faceids] = v2fent;
            int loopbase = outprim->loops.size();
            std::map<int, std::vector<int>> lut;
            std::map<int, int> vid2f;
            for (int ff = 0; ff < faceids.size(); ff++) {
                int f = faceids[ff];
                auto [start, len] = prim->polys[f];
                if (len < 2) {
                    log_warn("polygon has {} edges < 2", len);
                    return;
                }
                int resl = -1;
                for (int l = 0; l < len; l++) {
                    if (prim->loops[start + l] == vid) {
                        resl = l;
                        break;
                    }
                }
                if (resl == -1) {
                    log_warn("cannot find vertex {} in face {}", vid, f);
                    return;
                }
                auto vnext = prim->loops[start + (resl + 1) % len];
                auto vprev = prim->loops[start + (resl - 1 + len) % len];
                lut[vnext].push_back(vprev);
                if (vnext != vprev)
                    lut[vprev].push_back(vnext);
                vid2f.emplace(vnext, f);
            }
            //ZENO_P(lut);

            std::set<int> visited;
            auto dfs = [&] (auto &dfs, int vv0) -> void {
                if (visited.count(vv0)) return;
                visited.insert(vv0);
                auto vid2fit = vid2f.find(vv0);
                if (vid2fit == vid2f.end()) return;
                auto f0 = vid2fit->second;
                //ZENO_P(f0);
                outprim->loops.push_back(f0);
                auto lutit = lut.find(vv0);
                if (lutit == lut.end()) {
                    log_warn("lut lookup failed at {}", vv0);
                    return;
                }
                auto const &ffs = lutit->second;
                if (ffs.size() < 2) return;
                if (ffs.size() > 2) {
                    log_warn("edge shared by {} faces > 2", ffs.size());
                    return;
                }
                int vv1 = ffs[0];
                int vv2 = ffs[1];
                dfs(dfs, vv1);
                dfs(dfs, vv2);
            };
            dfs(dfs, lut.begin()->first);

            outprim->polys.emplace_back(loopbase, outprim->loops.size() - loopbase);
        });

        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimDualMesh, {
    {
    {"PrimitiveObject", "prim"},
    {"bool", "polygonate", "1"},
    {"bool", "keepBounds", "0"},
    //{"enum faces lines", "faceType", "faces"},
    //{"bool", "copyFaceAttrs", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
