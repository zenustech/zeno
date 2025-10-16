#include <zeno/para/parallel_for.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/core/NodeImpl.h>
#include <zeno/zeno.h>
#include <limits>

namespace zeno {
namespace {

struct PrimSmooth : INode {
    virtual void apply() override {
        auto prim = ZImpl(get_input<PrimitiveObject>("prim"));
        throw makeError("WIP");
        ZImpl(set_output("prim", std::move(prim)));
    }
};

ZENDEFNODE(PrimSmooth, {
    {
    {gParamType_Primitive, "prim", "", zeno::Socket_ReadOnly},
    },
    {
    {gParamType_Primitive, "prim"},
    },
    {
    },
    {"primitive"},
});

}
}
