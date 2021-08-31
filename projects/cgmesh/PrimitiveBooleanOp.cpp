#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include "EigenUtils.h"

namespace {

using namespace zeno;


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");
        auto op_type = get_param<std::string>("op_type");

        auto [VA, FA] = prim_to_eigen(primA.get());
        auto [VB, FB] = prim_to_eigen(primB.get());

        igl::MeshBooleanType boolean_type;
        if (op_type == "Union") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
        } else if (op_type == "Intersect") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
        } else if (op_type == "Minus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
        } else if (op_type == "RevMinus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
          std::swap(VA, VB); std::swap(FA, FB);
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
        igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, boolean_type, VC, FC, J);

        /*auto attrName = get_param<std::string>("attrName");
        printf("5\n");
        if (attrName.size()) {
            auto attrValA = get_input<NumericObject>("attrValA")->value;
            auto attrValB = get_input<NumericObject>("attrValB")->value;

            std::visit([&] (auto valA) {
                auto valB = std::get<decltype(valA)>(attrValB);
            }, attrValA);
        }*/

        auto primC = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VC, FC, primC.get());

        set_output("primC", std::move(primC));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "primA", "primB",
    {"float", "attrValA", "0"},
    {"float", "attrValB", "1"},
    },
    {
    "primC",
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "union"},
    {"string", "attrName", ""},
    },
    {"cgmesh"},
});

struct PrimitiveListBoolOp : INode {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");
        auto op_type = get_param<std::string>("op_type");

        auto [VA, FA] = prim_to_eigen(primA.get());
        auto [VB, FB] = prim_to_eigen(primB.get());

        igl::MeshBooleanType boolean_type;
        if (op_type == "Union") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
        } else if (op_type == "Intersect") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
        } else if (op_type == "Minus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
        } else if (op_type == "RevMinus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
          std::swap(VA, VB); std::swap(FA, FB);
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
        igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, boolean_type, VC, FC, J);

        /*auto attrName = get_param<std::string>("attrName");
        printf("5\n");
        if (attrName.size()) {
            auto attrValA = get_input<NumericObject>("attrValA")->value;
            auto attrValB = get_input<NumericObject>("attrValB")->value;

            std::visit([&] (auto valA) {
                auto valB = std::get<decltype(valA)>(attrValB);
            }, attrValA);
        }*/

        auto primC = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VC, FC, primC.get());

        set_output("primC", std::move(primC));
    }
}

ZENO_DEFNODE(PrimitiveListBoolOp)({
    {
    "primA", "primListB",
    {"float", "attrValA", "0"},
    {"float", "attrValB", "1"},
    },
    {
    "primListC",
    },
    {
    {"enum Union Intersect Minus RevMinus XOR Resolve", "op_type", "union"},
    {"string", "attrName", ""},
    },
    {"cgmesh"},
});

}
