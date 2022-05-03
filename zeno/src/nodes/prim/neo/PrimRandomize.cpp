#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#include <random>

namespace zeno {

ZENO_API void primRandomize(PrimitiveObject *prim, vec3f const &scale, int seed) {
#pragma omp parallel for
    for (int i = 0; i < prim->verts.size(); i++) {
        wangsrng rng(seed, i);
        vec3f offs{rng.next_float(), rng.next_float(), rng.next_float()};
        prim->verts[i] += (offs * 2 - 1) * scale;
    }
}

namespace {

struct PrimRandomize : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto scale = get_input2<vec3f>("scale");
        auto seed = get_input2<int>("seed");
        primRandomize(prim.get(), scale, seed);
        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimRandomize, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "scale", "1,1,1"},
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
