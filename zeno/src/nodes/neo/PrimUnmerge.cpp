#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/para/parallel_reduce.h>
#include <zeno/para/parallel_for.h>

namespace zeno {

ZENO_API std::vector<std::shared_ptr<PrimitiveObject>> primUnmergeVerts(PrimitiveObject *prim, std::string tagAttr) {
    if (!prim->verts.size()) return {};

    auto const &tagArr = prim->verts.attr<int>(tagAttr);
    int tagMax = parallel_reduce_max(tagArr.begin(), tagArr.end()) + 1;

    std::vector<std::shared_ptr<PrimitiveObject>> primList(tagMax);
    for (int tag = 0; tag < tagMax; tag++) {
        primList[tag] = std::make_shared<PrimitiveObject>();
    }

#if 1
    for (int tag = 0; tag < tagMax; tag++) {
        primList[tag]->assign(prim);
        primFilterVerts(primList[tag].get(), tagAttr, tag);
    }

#else
    std::vector<std::vector<int>> vert_revamp(tagMax);
    std::vector<int> vert_unrevamp(prim->verts.size());

    for (size_t i = 0; i < prim->verts.size(); i++) {
        int tag = tagArr[i];
        vert_revamp[tag].push_back(i);
        vert_unrevamp[i] = tag;
    }

    for (int tag = 0; tag < tagMax; tag++) {
        auto &revamp = vert_revamp[tag];
        auto const &outprim = primList[tag];

        outprim->verts.resize(revamp.size());
        parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
            outprim->verts[i] = prim->verts[revamp[i]];
        });
        prim->verts.foreach_attr([&] (auto const &key, auto const &inarr) {
            using T = std::decay_t<decltype(inarr[0])>;
            auto &outarr = outprim->verts.add_attr<T>(key);
            parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                outarr[i] = inarr[revamp[i]];
            });
        });
    }

    std::vector<std::vector<int>> face_revamp;

    auto mock = [&] (auto getter) {
        auto &prim_tris = getter(prim);
        if (prim_tris.size()) {
            face_revamp.clear();
            face_revamp.resize(tagMax);
            using T = std::decay_t<decltype(prim_tris[0])>;

            for (size_t i = 0; i < prim_tris.size(); i++) {
                auto ind = reinterpret_cast<decay_vec_t<T> const *>(&prim_tris[i]);
                int tag = vert_unrevamp[ind[0]];
                bool bad = false;
                for (int j = 1; j < is_vec_n<T>; j++) {
                    int new_tag = vert_unrevamp[ind[j]];
                    if (tag != new_tag) {
                        bad = true;
                        break;
                    }
                }
                if (!bad) face_revamp[tag].push_back(i);
            }

            for (int tag = 0; tag < tagMax; tag++) {
                auto &revamp = face_revamp[tag];
                auto &v_revamp = vert_revamp[tag];
                auto *outprim = primList[tag].get();
                auto &outprim_tris = getter(outprim);

                outprim_tris.resize(revamp.size());
                parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                    auto ind = reinterpret_cast<decay_vec_t<T> const *>(&prim_tris[revamp[i]]);
                    auto outind = reinterpret_cast<decay_vec_t<T> *>(&outprim_tris[i]);
                    for (int j = 0; j < is_vec_n<T>; j++) {
                        outind[j] = v_revamp[ind[j]];
                    }
                });

                prim_tris.foreach_attr([&] (auto const &key, auto const &inarr) {
                    using T = std::decay_t<decltype(inarr[0])>;
                    auto &outarr = outprim_tris.template add_attr<T>(key);
                    parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                        outarr[i] = inarr[revamp[i]];
                    });
                });
            }
        }
    };
    mock([] (auto &&p) -> auto & { return p->points; });
    mock([] (auto &&p) -> auto & { return p->lines; });
    mock([] (auto &&p) -> auto & { return p->tris; });
    mock([] (auto &&p) -> auto & { return p->quads; });

    if (prim->polys.size()) {
        face_revamp.clear();
        face_revamp.resize(tagMax);

        for (size_t i = 0; i < prim->polys.size(); i++) {
            auto &[base, len] = prim->polys[i];
            if (len <= 0) continue;
            int tag = vert_unrevamp[prim->loops[base]];
            bool bad = false;
            for (int j = base + 1; j < base + len; i++) {
                int new_tag = vert_unrevamp[prim->loops[j]];
                if (tag != new_tag) {
                    bad = true;
                    break;
                }
            }
            if (!bad) face_revamp[tag].push_back(i);
        }

        for (int tag = 0; tag < tagMax; tag++) {
            auto &revamp = face_revamp[tag];
            auto &v_revamp = vert_revamp[tag];
            auto *outprim = primList[tag].get();

            outprim->polys.resize(revamp.size());
            for (size_t i = 0; i < revamp.size(); i++) {
                auto const &[base, len] = prim->polys[revamp[i]];
                int new_base = outprim->loops.size();
                for (int j = base; j < base + len; j++) {
                    outprim->loops.push_back(prim->loops[j]);
                }
                outprim->polys[i] = {new_base, len};
            }

            prim->polys.foreach_attr([&] (auto const &key, auto const &inarr) {
                using T = std::decay_t<decltype(inarr[0])>;
                auto &outarr = outprim->polys.add_attr<T>(key);
                parallel_for((size_t)0, revamp.size(), [&] (size_t i) {
                    outarr[i] = inarr[revamp[i]];
                });
            });
        }
    }
#endif

    return primList;
}

namespace {

struct PrimUnmerge : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto method = get_input<StringObject>("method")->get();

        auto primList = primUnmergeVerts(prim.get(), tagAttr);

        auto listPrim = std::make_shared<ListObject>();
        for (auto &primPtr: primList) {
            listPrim->arr.push_back(std::move(primPtr));
        }
        set_output("listPrim", std::move(listPrim));
    }
};

ZENDEFNODE(PrimUnmerge, {
    {
        {"primitive", "prim"},
        {"string", "tagAttr", "tag"},
        {"enum verts faces", "method", "verts"},
    },
    {
        {"list", "listPrim"},
    },
    {
    },
    {"primitive"},
});

}
}
