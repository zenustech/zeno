#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <cmath>

namespace zeno {
namespace {

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
