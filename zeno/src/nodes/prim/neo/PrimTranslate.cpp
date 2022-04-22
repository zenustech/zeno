#include "zeno/types/StringObject.h"
#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>

namespace zeno {

ZENO_API void primTranslate(PrimitiveObject *prim, vec3f const &offset) {
#pragma omp parallel for
    for (int i = 0; i < prim->verts.size(); i++) {
        prim->verts[i] = prim->verts[i] + offset;
    }
}

ZENO_API void primScale(PrimitiveObject *prim, vec3f const &scale) {
#pragma omp parallel for
    for (int i = 0; i < prim->verts.size(); i++) {
        prim->verts[i] = prim->verts[i] * scale;
    }
}

namespace {

struct PrimTranslate : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto offset = get_input2<vec3f>("offset");
        primTranslate(prim.get(), offset);
        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimTranslate, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "offset", "0,0,0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimScale : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto scale = get_input2<vec3f>("scale");
        primScale(prim.get(), scale);
        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimScale, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "scale", "1,1,1"},
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
