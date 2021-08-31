#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include "EigenUtils.h"

namespace zeno {
// defined in PrimitiveMeshingFix.cpp:
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> prim_to_eigen_with_fix(PrimitiveObject const *primA);
}

namespace {

using namespace zeno;


struct PrimitiveBooleanOp : INode {
    auto boolean_op(Eigen::MatrixXd const *pVA, Eigen::MatrixXi const *pFA,
            PrimitiveObject const *primA, PrimitiveObject const *primB) {
        auto [VB, FB] = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primB) : prim_to_eigen(primB);
        auto const *pVB = &VB;
        auto const *pFB = &FB;

        auto op_type = get_param<std::string>("op_type");
        igl::MeshBooleanType boolean_type;
        if (op_type == "Union") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
        } else if (op_type == "Intersect") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
        } else if (op_type == "Minus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
        } else if (op_type == "RevMinus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
          std::swap(pVA, pVB); std::swap(pFA, pFB);
        } else if (op_type == "XOR") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_XOR;
        } else if (op_type == "Resolve") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_RESOLVE;
        } else {
          throw Exception("bad boolean op type: " + op_type);
        }

        Eigen::MatrixXd VC;
        Eigen::MatrixXi FC;
        Eigen::VectorXi J;
        igl::copyleft::cgal::mesh_boolean(*pVA, *pFA, *pVB, *pFB, boolean_type, VC, FC, J);

        auto primC = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VC, FC, primC.get());

        if (get_param<bool>("assignAttrs")) {
            for (auto const &[key, arrA]: primA->m_attrs) {
                if (key == "pos") continue;
                if (!primB->has_attr(key)) continue;
                std::visit([&] (auto const &arrA) {
                    using T = std::decay_t<decltype(arrA[0])>;
                    if (!primB->attr_is<T>(key)) return;
                    auto &arrB = primB->attr<T>(key);
                    auto &arrC = primC->add_attr<T>(key);
                    for (int i = 0; i < primC->size(); i++) {
                        int j = J(i), jmax = pFA->rows();
                        if (j < jmax) {
                            arrC[i] = arrA[j];
                        } else {
                            arrC[i] = arrB[j - jmax];
                        }
                    }
                }, arrA);
            }
        }

        return primC;
    }

    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");

        auto [VA, FA] = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primA.get()) : prim_to_eigen(primA.get());
        auto primC = boolean_op(&VA, &FA, primA.get(), primB.get());

        set_output("primC", std::move(primC));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "primA", "primB",
    },
    {
    "primC",
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "Union"},
    {"bool", "assignAttrs", "1"},
    {"bool", "doMeshFix", "1"},
    },
    {"cgmesh"},
});

#if 1
struct PrimitiveListBoolOp : PrimitiveBooleanOp {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primListB = get_input<ListObject>("primListB");

        auto [VA, FA] = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primA.get()) : prim_to_eigen(primA.get());

        auto listB = primListB->get<std::shared_ptr<PrimitiveObject>>();
        auto primListC = std::make_shared<ListObject>();
        primListC->arr.resize(listB.size());
        //#pragma omp parallel for
        for (int i = 0; i < listB.size(); i++) {
            printf("PrimitiveListBoolOp: processing mesh #%d...\n", i);
            auto const &primB = listB[i];
            auto primC = boolean_op(&VA, &FA, primA.get(), primB.get());
            primListC->arr[i] = std::move(primC);
        }

        set_output("primListC", std::move(primListC));
    }
};

ZENO_DEFNODE(PrimitiveListBoolOp)({
    {
    "primA", "primListB",
    },
    {
    "primListC",
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "union"},
    {"bool", "assignAttrs", "1"},
    {"bool", "doMeshFix", "1"},
    },
    {"cgmesh"},
});
#endif

}
