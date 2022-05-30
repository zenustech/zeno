#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/para/parallel_for.h>
#include <zeno/para/task_group.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

ZENO_API std::shared_ptr<PrimitiveObject> primDuplicate(PrimitiveObject *parsPrim, PrimitiveObject *meshPrim, std::string dirAttr, std::string tanAttr, std::string radAttr, float radius, int seed) {
    auto prim = std::make_shared<PrimitiveObject>();
    auto hasDirAttr = boolean_variant(!dirAttr.empty());
    auto hasRadAttr = boolean_variant(!radAttr.empty());
    auto hasRadius = boolean_variant(radius != 1);

    task_group tg;
    prim->verts.resize(parsPrim->verts.size() * meshPrim->verts.size());
    tg.add([&] {
        std::visit([&] (auto hasDirAttr, auto hasRadius, auto hasRadAttr) {
            auto func = [&] (auto const &accRad) {
                auto func = [&] (auto const &accDir, auto hasTanAttr, auto const &accTan) {
                    parallel_for((size_t)0, parsPrim->verts.size(), [&] (size_t i) {
                        auto basePos = parsPrim->verts[i];
                        for (size_t j = 0; j < meshPrim->verts.size(); j++) {
                            auto pos = meshPrim->verts[j];
                            if constexpr (hasRadAttr) {
                                pos *= accRad[i];
                            }
                            if constexpr (hasRadius) {
                                pos *= radius;
                            }
                            if constexpr (hasDirAttr) {
                                auto t0 = accDir[i];
                                vec3f t1, t2;
                                if constexpr (hasTanAttr) {
                                    t1 = accTan[i];
                                    t2 = cross(t0, t1);
                                } else {
                                    pixarONB(t0, t1, t2);
                                }
                                pos = pos[0] * t0 + pos[1] * t1 + pos[2] * t2;
                            }
                            prim->verts[i * meshPrim->verts.size() + j] = basePos + pos;
                        }
                    });
                };
                if constexpr (hasDirAttr) {
                    auto const &accDir = meshPrim->attr<vec3f>(dirAttr);
                    if (!tanAttr.empty())
                        func(accDir, std::true_type{}, meshPrim->attr<vec3f>(tanAttr));
                    else
                        func(accDir, std::false_type{}, std::array<int, 0>{});
                } else {
                    func(std::array<int, 0>{}, std::false_type{}, std::array<int, 0>{});
                }
            };
            if constexpr (hasRadAttr)
                meshPrim->verts.attr_visit(radAttr, func);
            else
                func(std::array<int, 0>{});
        }, hasDirAttr, hasRadius, hasRadAttr);
    });

    auto func = [&] (auto &primAttrs, auto &meshAttrs, auto &parsAttrs) {
        meshAttrs.foreach_attr([&] (auto const &key, auto const &arrMesh) {
            using T = std::decay_t<decltype(arrMesh[0])>;
            primAttrs.template add_attr<T>(key);
            tg.add([&] {
                auto &arrOut = primAttrs.template attr<T>(key);
                parallel_for((size_t)0, parsAttrs.size(), [&] (size_t i) {
                    for (size_t j = 0; j < meshAttrs.size(); j++) {
                        arrOut[i * meshAttrs.size() + j] = arrMesh[j];
                    }
                });
            });
        });
        parsAttrs.foreach_attr([&] (auto const &key, auto const &arrPars) {
            if (meshAttrs.has_attr(key)) return;
            using T = std::decay_t<decltype(arrPars[0])>;
            primAttrs.template add_attr<T>(key);
            tg.add([&] {
                auto &arrOut = primAttrs.template attr<T>(key);
                parallel_for((size_t)0, arrPars.size(), [&] (size_t i) {
                    auto value = arrPars[i];
                    for (size_t j = 0; j < meshAttrs.size(); j++) {
                        arrOut[i * meshAttrs.size() + j] = value;
                    }
                });
            });
        });
    };
    func(prim->verts, meshPrim->verts, parsPrim->verts);
    func(prim->points, meshPrim->points, parsPrim->points);
    func(prim->lines, meshPrim->lines, parsPrim->lines);
    func(prim->tris, meshPrim->tris, parsPrim->tris);
    func(prim->quads, meshPrim->quads, parsPrim->quads);
    func(prim->polys, meshPrim->polys, parsPrim->polys);
    func(prim->loops, meshPrim->loops, parsPrim->loops);

    tg.run();

    return prim;
}

namespace {

struct PrimDuplicate : INode {
    virtual void apply() override {
        auto parsPrim = get_input<PrimitiveObject>("parsPrim");
        auto meshPrim = get_input<PrimitiveObject>("meshPrim");
        auto tanAttr = get_input2<std::string>("tanAttr");
        auto dirAttr = get_input2<std::string>("dirAttr");
        auto radAttr = get_input2<std::string>("radAttr");
        auto radius = get_input2<float>("radius");
        auto seed = get_input2<int>("seed");
        auto prim = primDuplicate(parsPrim.get(), meshPrim.get(),
                                  dirAttr, tanAttr, radAttr, radius, seed);
        set_output("prim", prim);
    }
};

ZENDEFNODE(PrimDuplicate, {
    {
    {"PrimitiveObject", "parsPrim"},
    {"PrimitiveObject", "meshPrim"},
    {"string", "dirAttr", ""},
    {"string", "tanAttr", ""},
    {"string", "radAttr", ""},
    {"float", "radius", "1"},
    {"int", "seed", "0"},
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
