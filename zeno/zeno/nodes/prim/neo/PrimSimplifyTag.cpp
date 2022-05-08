#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/random.h>
#include <unordered_map>

namespace zeno {
namespace {

static void primSimplifyTag(PrimitiveObject *prim, std::string tagAttr) {
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

static void primColorByTag(PrimitiveObject *prim, std::string tagAttr, std::string clrAttr) {
    auto const &tag = prim->verts.attr<int>(tagAttr);
    auto &clr = prim->verts.add_attr<vec3f>(clrAttr);
    std::unordered_map<int, vec3f> lut;
    for (int i = 0; i < tag.size(); i++) {
        auto k = tag[i];
        auto it = lut.find(k);
        if (it != lut.end()) {
            clr[i] = it->second;
        } else {
            vec3f c(frand(), frand(), frand());
            c = 0.4f + 0.6f * c;
            lut.emplace(k, c);
            clr[i] = c;
        }
    }
}

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
