#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <unordered_map>
#include <random>

namespace zeno {

ZENO_API void primSimplifyTag(PrimitiveObject *prim, std::string tagAttr) {
    auto &tag = prim->verts.attr<int>(tagAttr);
    std::unordered_map<int, int> lut;
    int top = 0;
    for (int i = 0; i < tag.size(); i++) {
        auto k = tag[i];
        auto it = lut.find(k);
        if (it != lut.end()) {
            tag[i] = it->second;
        } else {
            int c = top++;
            lut.emplace(k, c);
            tag[i] = c;
        }
    }
}

ZENO_API void primColorByTag(PrimitiveObject *prim, std::string tagAttr, std::string clrAttr) {
    auto const &tag = prim->verts.attr<int>(tagAttr);
    auto &clr = prim->verts.add_attr<vec3f>(clrAttr);
    std::unordered_map<int, vec3f> lut;
    std::mt19937 gen;
    std::uniform_real_distribution<float> unif(0.4f, 1.0f);
    for (int i = 0; i < tag.size(); i++) {
        auto k = tag[i];
        auto it = lut.find(k);
        if (it != lut.end()) {
            clr[i] = it->second;
        } else {
            vec3f c(unif(gen), unif(gen), unif(gen));
            lut.emplace(k, c);
            clr[i] = c;
        }
    }
}


namespace {

struct PrimSimplifyTag : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();

        primSimplifyTag(prim.get(), tagAttr);

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimSimplifyTag, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimColorByTag : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto tagAttr = get_input<StringObject>("tagAttr")->get();
        auto clrAttr = get_input<StringObject>("clrAttr")->get();

        primColorByTag(prim.get(), tagAttr, clrAttr);

        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimColorByTag, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "tagAttr", "tag"},
    {"string", "clrAttr", "clr"},
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
