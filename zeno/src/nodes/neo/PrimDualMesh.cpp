#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/utils/arrayindex.h>
#include <cmath>

namespace zeno {
namespace {

struct PrimDualMesh : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceType = get_input2<std::string>("faceType");
        auto copyFaceAttrs = get_input2<bool>("copyFaceAttrs");
        auto outprim = std::make_shared<PrimitiveObject>();
        set_output("prim", std::move(outprim));
    }
};

ZENDEFNODE(PrimDualMesh, {
    {
    {"PrimitiveObject", "prim"},
    {"enum faces lines", "faceType", "faces"},
    {"bool", "copyFaceAttrs", "1"},
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
