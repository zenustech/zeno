#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/wangsrng.h>
#include <zeno/utils/perlin.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/para/parallel_for.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>
#include <cstring>
#include <cstdlib>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

ZENO_API void primPerlinNoise(PrimitiveObject *prim, std::string inAttrName, std::string attrName, std::string randType, std::string combType, float scale, int seed) {
    auto randty = enum_variant<RandTypes>(array_index(lutRandTypes, randType));
    auto combIsAdd = boolean_variant(combType == "add");
    std::visit([&] (auto &&randty, auto &&combIsAdd) {
        using T = std::invoke_result_t<decltype(randty), wangsrng &>;
        auto &arr = prim->add_attr<T>(attrName);
        parallel_for((size_t)0, arr.size(), [&] (size_t i) {
            wangsrng rng(seed, i);
            T offs = randty(rng) * scale;
            if (combIsAdd.value)
                arr[i] += offs;
            else
                arr[i] = offs;
        });
    }, randty, combIsAdd);
}

namespace {

struct PrimPerlinNoise : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto scale = get_input2<float>("scale");
        auto seed = get_input2<int>("seed");
        auto attrName = get_input2<std::string>("attr");
        auto inAttrName = get_input2<std::string>("inAttr");
        auto randType = get_input2<std::string>("randType");
        auto combType = get_input2<std::string>("combType");
        primPerlinNoise(prim.get(), attrName, randType, combType, scale, seed);
        set_output("prim", get_input("prim"));
    }
};

ZENDEFNODE(PrimPerlinNoise, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "inAttr", "pos"},
    {"string", "attr", "clr"},
    {"float", "scale", "1"},
    {"float", "detail", "1"},
    {"float", "roughness", "1"},
    {"float", "disortion", "1"},
    {"int", "seed", "0"},
    {"enum set add", "combType", "add"},
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
