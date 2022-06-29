#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#define MESHFIX_WITH_EIGEN
#include "libigl/meshfix/meshfix.h"
#include "EigenUtils.h"

namespace zeno {
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> prim_to_eigen_with_fix(PrimitiveObject const *primA) {
    auto [VA, FA] = prim_to_eigen(primA);
    Eigen::MatrixXd VB;
    Eigen::MatrixXi FB;
    meshfix(VA, FA, VB, FB);
    return {std::move(VB), std::move(FB)};
}
}

namespace {
using namespace zeno;

struct PrimitiveMeshingFix : INode {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("prim");

        auto [VB, FB] = prim_to_eigen_with_fix(primA.get());

        auto primFixed = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VB, FB, primFixed.get());

        set_output("primFixed", std::move(primFixed));
    }
};

ZENO_DEFNODE(PrimitiveMeshingFix)({
    {
    "prim",
    },
    {
    "primFixed",
    },
    {
    },
    {"cgmesh"},
});

}
