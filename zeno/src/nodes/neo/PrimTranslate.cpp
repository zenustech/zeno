#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

ZENO_API void primTranslate(PrimitiveObject *prim, vec3f const &offset) {
    parallel_for((size_t)0, prim->verts.size(), [&] (size_t i) {
        prim->verts[i] = prim->verts[i] + offset;
    });
}

ZENO_API void primScale(PrimitiveObject *prim, vec3f const &scale) {
    parallel_for((size_t)0, prim->verts.size(), [&] (size_t i) {
        prim->verts[i] = prim->verts[i] * scale;
    });
}

namespace {

struct PrimTranslate : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto offset = ZImpl(get_input2<zeno::vec3f>("offset"));
        primTranslate(prim.get(), offset);
        ZImpl(set_output("prim", ZImpl(clone_input("prim"))));
    }
};

ZENDEFNODE(PrimTranslate, {
                              {
                                  {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
                                  {gParamType_Vec3f, "offset", "0,0,0"},
                              },
                              {
                                  {gParamType_Primitive, "prim"},
                              },
                              {},
                              {"primitive"},
                          });

struct PrimScale : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        auto scale = ZImpl(get_input2<zeno::vec3f>("scale"));
        primScale(prim.get(), scale);
        ZImpl(set_output("prim", ZImpl(clone_input("prim"))));
    }
};

ZENDEFNODE(PrimScale, {
                          {
                              {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
                              {gParamType_Vec3f, "scale", "1,1,1"},
                          },
                          {
                              {gParamType_Primitive, "prim"},
                          },
                          {},
                          {"primitive"},
                      });

} // namespace
} // namespace zeno
