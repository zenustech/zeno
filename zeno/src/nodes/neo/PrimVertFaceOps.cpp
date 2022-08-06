#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <cmath>

namespace zeno {
namespace {

template <class T>
struct meth_sum {
    T value{0};

    void add(T x) {
        value += x;
    }

    T get() {
        return value;
    }
};

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

template <class T>
struct meth_min {
    T value{std::numeric_limits<decay_vec_t<T>>::max()};

    void add(T x) {
        value = zeno::min(value, x);
    }

    T get() {
        return value;
    }
};

template <class T>
struct meth_max {
    T value{std::numeric_limits<decay_vec_t<T>>::min()};

    void add(T x) {
        value = zeno::max(value, x);
    }

    T get() {
        return value;
    }
};

struct face_lines {
    static auto &from_prim(PrimitiveObject *prim) {
        return prim->lines;
    }

    template <class Func>
    static void foreach_ind(PrimitiveObject *, vec2i const &ind, Func const &func) {
        func(ind[0]);
        func(ind[1]);
    }
};

struct face_tris {
    static auto &from_prim(PrimitiveObject *prim) {
        return prim->tris;
    }

    template <class Func>
    static void foreach_ind(PrimitiveObject *, vec3i const &ind, Func const &func) {
        func(ind[0]);
        func(ind[1]);
        func(ind[2]);
    }
};

struct face_quads {
    static auto &from_prim(PrimitiveObject *prim) {
        return prim->quads;
    }

    template <class Func>
    static void foreach_ind(PrimitiveObject *, vec4i const &ind, Func const &func) {
        func(ind[0]);
        func(ind[1]);
        func(ind[2]);
        func(ind[3]);
    }
};

struct face_polys {
    static auto &from_prim(PrimitiveObject *prim) {
        return prim->polys;
    }

    template <class Func>
    static void foreach_ind(PrimitiveObject *prim, vec2i const &ind, Func const &func) {
        auto [base, len] = ind;
        for (int l = base; l < base + len; l++) {
            func(prim->loops[l]);
        }
    }
};

struct PrimVertsAttrToFaces : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceType = get_input2<std::string>("faceType");
        auto attr = get_input2<std::string>("vertAttr");
        auto attrOut = get_input2<std::string>("faceAttr");
        auto method = get_input2<std::string>("method");

        std::visit([&] (auto faceTy) {
            auto &prim_faces = faceTy.from_prim(prim.get());

            prim->verts.attr_visit(attr, [&] (auto const &vertsArr) {
                using T = std::decay_t<decltype(vertsArr[0])>;
                auto &facesArr = prim_faces.template add_attr<T>(attrOut);

                std::visit([&] (auto reducerTy) {
                    for (int i = 0; i < prim_faces.size(); i++) {
                        decltype(reducerTy) reducer;
                        faceTy.foreach_ind(prim.get(), prim_faces[i], [&] (int ind) {
                            reducer.add(vertsArr[ind]);
                        });
                        facesArr[i] = reducer.get();
                    }
                }, enum_variant<std::variant<
                    meth_sum<T>, meth_average<T>, meth_min<T>, meth_max<T>
                >>(array_index({
                    "sum", "average", "min", "max"
                }, method)));
            });

        }, enum_variant<std::variant<
            face_lines, face_tris, face_quads, face_polys
        >>(array_index({
            "lines", "tris", "quads", "polys"
        }, faceType)));

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimVertsAttrToFaces, {
    {
    {"PrimitiveObject", "prim"},
    {"enum lines tris quads polys", "faceType", "tris"},
    {"string", "vertAttr", "tmp"},
    {"string", "faceAttr", "tmp"},
    {"enum sum average min max", "method", "average"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimFacesAttrToVerts : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceType = get_input2<std::string>("faceType");
        auto attr = get_input2<std::string>("faceAttr");
        auto attrOut = get_input2<std::string>("vertAttr");
        auto method = get_input2<std::string>("method");
        auto deflValPtr = get_input<NumericObject>("deflVal");

        std::visit([&] (auto faceTy) {
            auto &prim_faces = faceTy.from_prim(prim.get());

            prim_faces.attr_visit(attr, [&] (auto const &facesArr) {
                using T = std::decay_t<decltype(facesArr[0])>;
                auto &vertsArr = prim->verts.add_attr<T>(attrOut);

                std::vector<std::vector<int>> v2f(prim->verts.size());
                for (int i = 0; i < prim_faces.size(); i++) { // todo: parallel_push_back_multi
                    faceTy.foreach_ind(prim.get(), prim_faces[i], [&] (int ind) {
                        v2f[ind].push_back(i);
                    });
                }

                auto deflVal = deflValPtr->get<T>();
                std::visit([&] (auto reducerTy) {
                    for (int i = 0; i < prim->verts.size(); i++) {
                        if (v2f[i].empty()) {
                            vertsArr[i] = deflVal;
                        } else {
                            decltype(reducerTy) reducer;
                            for (auto l: v2f[i]) {
                                reducer.add(facesArr[l]);
                            }
                            vertsArr[i] = reducer.get();
                        }
                    }
                }, enum_variant<std::variant<
                    meth_sum<T>, meth_average<T>, meth_min<T>, meth_max<T>
                >>(array_index({
                    "sum", "average", "min", "max"
                }, method)));
            });
        }, enum_variant<std::variant<
            face_lines, face_tris, face_quads, face_polys
        >>(array_index({
            "lines", "tris", "quads", "polys"
        }, faceType)));

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFacesAttrToVerts, {
    {
    {"PrimitiveObject", "prim"},
    {"enum lines tris quads polys", "faceType", "tris"},
    {"string", "faceAttr", "tmp"},
    {"string", "vertAttr", "tmp"},
    {"float", "deflVal", "0"},
    {"enum sum average min max", "method", "average"},
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
