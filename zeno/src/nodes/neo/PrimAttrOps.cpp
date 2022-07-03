#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <cmath>

namespace zeno {
namespace {

struct PrimFillAttr : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto value = get_input<NumericObject>("value");
        auto attr = get_input2<std::string>("attr");
        auto type = get_input2<std::string>("type");
        std::visit([&] (auto ty) {
            using T = decltype(ty);
            auto &arr = prim->verts.add_attr<T>(attr);
            auto val = value->get<T>();
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = val;
            }
        }, enum_variant<std::variant<
            float, vec3f, int
        >>(array_index({
            "float", "vec3f", "int"
        }, type)));
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFillAttr, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "tag"},
    {"enum float vec3f int", "type"},
    {"float", "value", "0"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

struct PrimFloatAttrToInt : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto attr = get_input2<std::string>("attr");
        auto attrOut = get_input2<std::string>("attrOut");
        auto &inArr = prim->verts.attr<float>(attr);
        if (attrOut == attr) {
            std::vector<int> outArr;
            for (size_t i = 0; i < inArr.size(); i++) {
                outArr[i] = std::rint(inArr[i] + 0.5f);
            }
            prim->verts.add_attr<int>(attrOut) = std::move(outArr);
        } else {
            prim->verts.attrs.erase(attrOut);
            auto &outArr = prim->verts.add_attr<int>(attrOut);
            for (size_t i = 0; i < inArr.size(); i++) {
                outArr[i] = std::rint(inArr[i] + 0.5f);
            }
        }
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimFloatAttrToInt, {
    {
    {"PrimitiveObject", "prim"},
    {"string", "attr", "tag"},
    {"string", "attrOut", "tag"},
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
