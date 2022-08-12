#include <zeno/para/parallel_for.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveUtils.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/CurveObject.h>
#include <zeno/utils/arrayindex.h>
#include <zeno/utils/variantswitch.h>
#include <zeno/extra/TempNode.h>
#include <zeno/core/INode.h>
#include <zeno/zeno.h>
#include <limits>

namespace zeno {
namespace {

struct PrimSmooth : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        throw makeError("WIP");
        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(PrimSmooth, {
    {
    {"PrimitiveObject", "prim"},
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
