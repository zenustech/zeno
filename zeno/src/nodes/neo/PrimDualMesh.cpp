#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
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

template <class Func>
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
}

struct PrimDualMesh : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceType = get_input2<std::string>("faceType");
        auto copyFaceAttrs = get_input2<bool>("copyFaceAttrs");
        auto outprim = std::make_shared<PrimitiveObject>();

        //bool hasOverlappedEdge = false;
        //std::map<std::pair<int, int>, std::pair<int, int>> e2f;
        //prim_foreach_faces_edges(prim.get(), [&] (int face_i, auto const &foreach_edges) {
            //foreach_edges([&] (int v1, int v2) {
                //std::pair<int, int> key(std::min(v1, v2), std::max(v1, v2));
                //auto [it, succ] = e2f.try_emplace(key, face_i, -1);
                //if (!succ) {
                    //if (it->second.second == -1) // overlapped(l1, l2)
                        //hasOverlappedEdge = true;
                    //it->second.second = face_i;
                //}
            //});
        //});
        //if (hasOverlappedEdge)
            //log_warn("PrimDualMesh: got overlapped edge");

        std::map<int, std::vector<int>> v2f;
        outprim->verts.resize(prim->tris.size() + prim->quads.size() + prim->polys.size());
        prim_foreach_faces_corns(prim.get(), [&] (int face_i, auto const &foreach_corns) {
            meth_average<vec3f> reducer;
            foreach_corns([&] (int vert_i) {
                reducer.add(prim->verts[vert_i]);
                v2f[vert_i].push_back(face_i);
            });
            outprim->verts[face_i] = reducer.get();
        });

        //for (auto const &[edge_vs, faces]: e2f) {
            //auto [f1, f2] = faces;//thiscanonlygetedges...
        //}
        //ZENO_P(v2f);
        for (auto const &[vid, faceids]: v2f) {
            int loopbase = outprim->loops.size();
            for (auto f: faceids) {
                prim_face_query(prim.get(), f);
            }
            //outprim->loops->insert(outprim->loops.end(), faceids.begin(), faceids.end());
            outprim->polys.emplace_back(loopbase, faceids.size());
        }

        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimDualMesh, {
    {
    {"PrimitiveObject", "prim"},
    {"enum faces lines", "faceType", "faces"},
    {"bool", "copyFaceAttrs", "1"},
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
