#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include "EigenUtils.h"
#include "igl_sink.h"

namespace zeno {
// defined in PrimitiveMeshingFix.cpp:
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> prim_to_eigen_with_fix(PrimitiveObject const *primA);
}

namespace {

using namespace zeno;


struct PrimitiveBooleanOp : INode {
    auto boolean_op(Eigen::MatrixXd const &VA, Eigen::MatrixXi const &FA,
            PrimitiveObject const *primA, PrimitiveObject const *primB) {
        auto [VB, FB] = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primB) : prim_to_eigen(primB);

        Eigen::MatrixXd VC;
        Eigen::MatrixXi FC;
        Eigen::VectorXi J;
        auto op_type = get_param<std::string>("op_type");
        igl_mesh_boolean(VA, FA, VB, FB, op_type, VC, FC, J);

        auto primC = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VC, FC, primC.get());

        bool anyFromA = false, anyFromB = false;
        if (get_param<bool>("calcAnyFrom")) {
            for (int i = 0; i < primC->size(); i++) {
                int j = J(i), jmax = FA.rows();
                if (j < 0) printf("j=%d\n", j);
                if (j < jmax) {
                    anyFromA = true;
                } else {
                    anyFromB = true;
                }
            }
        }

        if (get_param<bool>("assignAttrs")) {
            for (auto const &[key, arrA]: primA->m_attrs) {
                if (key == "pos") continue;
                if (!primB->has_attr(key)) continue;
                std::visit([&, key = key] (auto const &arrA) {
                    using T = std::decay_t<decltype(arrA[0])>;
                    if (!primB->attr_is<T>(key)) return;
                    auto &arrB = primB->attr<T>(key);
                    auto &arrC = primC->add_attr<T>(key);
                    for (int i = 0; i < primC->size(); i++) {
                        int j = J(i), jmax = FA.rows();
                        if (j < jmax) {
                            arrC[i] = arrA[j];
                        } else {
                            arrC[i] = arrB[j - jmax];
                        }
                    }
                }, arrA);
            }
        }

        return std::make_tuple(primC, anyFromA, anyFromB);
    }

    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");

        auto [VA, FA] = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primA.get()) : prim_to_eigen(primA.get());
        auto [primC, anyFromA, anyFromB] = boolean_op(VA, FA, primA.get(), primB.get());

        set_output("primC", std::move(primC));
        set_output("anyFromA", std::make_shared<NumericObject>(anyFromA));
        set_output("anyFromB", std::make_shared<NumericObject>(anyFromB));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "primA", "primB",
    },
    {
    "primC", {"bool", "anyFromA"}, {"bool", "anyFromB"},
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
        std::vector<std::pair<bool, std::shared_ptr<PrimitiveObject>>> listC(listB.size());

        #pragma omp parallel for
        for (int i = 0; i < listB.size(); i++) {
            printf("PrimitiveListBoolOp: processing mesh #%d...\n", i);
            auto const &primB = listB[i];
            auto [primC, anyFromA, anyFromB] = boolean_op(VA, FA, primA.get(), primB.get());
            listC[i] = std::make_pair(anyFromA, std::move(primC));
        }

        auto primListAllFromB = std::make_shared<ListObject>();
        auto primListAnyFromA = std::make_shared<ListObject>();
        for (auto const &[anyFromA, primPtr]: listC) {
            if (anyFromA)
                primListAnyFromA->arr.push_back(primPtr);
            else
                primListAllFromB->arr.push_back(primPtr);
        }

        set_output("primListAllFromB", std::move(primListAllFromB));
        set_output("primListAnyFromA", std::move(primListAnyFromA));
    }
};

ZENO_DEFNODE(PrimitiveListBoolOp)({
    {
    "primA", "primListB",
    },
    {
    "primListAllFromB",
    "primListAnyFromA",
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "union"},
    {"bool", "assignAttrs", "1"},
    {"bool", "calcAnyFrom", "1"},
    {"bool", "doMeshFix", "1"},
    },
    {"cgmesh"},
});
#endif

}
