#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/logger.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include "EigenUtils.h"
#include "igl_sink.h"
#include <zeno/types/UserData.h>

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
            for (int i = 0; i < J.size(); i++) {
                if (J(i) < FA.rows()) {
                    anyFromA = true;
                } else {
                    anyFromB = true;
                }
            }
        }

        if (auto attrName = get_param<std::string>("faceAttrName"); attrName.size()) {
            auto attrValA = get_input<NumericObject>("faceAttrA")->value;
            auto attrValB = get_input<NumericObject>("faceAttrB")->value;
            std::visit([&] (auto const &valA) {
                using T = std::decay_t<decltype(valA)>;
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, vec3f>) {
                    auto valB = std::get<T>(attrValB);
                    auto &arrC = primC->tris.add_attr<T>(attrName);
                    for (int i = 0; i < primC->tris.size(); i++) {
                        if (J(i) < FA.rows()) {
                            arrC[i] = valA;
                        } else {
                            arrC[i] = valB;
                        }
                    }
                }
            }, attrValA);
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
    {"NumericObject", "faceAttrA"}, {"NumericObject", "faceAttrB"},
    },
    {
    "primC", {"bool", "anyFromA"}, {"bool", "anyFromB"},
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "Union"},
    {"string", "faceAttrName", ""},
    {"bool", "calcAnyFrom", "0"},
    {"bool", "doMeshFix", "1"},
    },
    {"cgmesh"},
});

#if 1
struct PrimitiveListBoolOp : PrimitiveBooleanOp {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primListB = get_input<ListObject>("primListB");

        auto VFA = get_param<bool>("doMeshFix") ? prim_to_eigen_with_fix(primA.get()) : prim_to_eigen(primA.get());

        auto listB = primListB->get<PrimitiveObject>();
        std::vector<std::pair<bool, std::shared_ptr<PrimitiveObject>>> listC(listB.size());

        #pragma omp parallel for
        for (int i = 0; i < listB.size(); i++) {
            log_debug("PrimitiveListBoolOp: processing mesh #{}...", i);
            auto const &primB = listB[i];
            auto [primC, anyFromA, anyFromB] = boolean_op(VFA.first, VFA.second, primA.get(), primB.get());
            listC[i] = std::make_pair(anyFromA, std::move(primC));
        }

        auto lutList = std::make_shared<ListObject>();
        auto primList = std::make_shared<ListObject>();
        lutList->arr.resize(listC.size());
        int lutcnt=-1;
        for (auto const &[anyFromA, primPtr]: listC) { lutcnt++;
            primPtr->userData().set("anyFromA", objectFromLiterial(anyFromA));
            if (get_param<bool>("noNullMesh") && primPtr->size() == 0) {
                auto cnt = std::make_shared<NumericObject>();
                cnt->set((int)-1);
                lutList->arr[lutcnt] = std::move(cnt);
                log_info("PrimListBool got null mesh {}", (void *)primPtr.get());
                continue;
            }
            auto cnt = std::make_shared<NumericObject>();
            cnt->set((int)primList->arr.size());
            lutList->arr[lutcnt] = std::move(cnt);
            primList->arr.push_back(primPtr);
        }

        set_output("primList", std::move(primList));
        set_output("lutList", std::move(lutList));
    }
};

ZENO_DEFNODE(PrimitiveListBoolOp)({
    {
    "primA", "primListB",
    },
    {
    "primList", "lutList",
    //{"bool", "anyFromA"}, {"bool", "anyFromB"},
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "Union"},
    {"string", "faceAttrName", ""},
    {"bool", "assignAttrs", "1"},
    {"bool", "calcAnyFrom", "0"},
    {"bool", "doMeshFix", "1"},
    {"bool", "noNullMesh", "1"},
    {"string", "DEPRECATED", ""},
    },
    {"cgmesh"},
});
#endif

}
