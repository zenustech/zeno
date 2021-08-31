#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <Eigen/Core>

namespace {
using namespace zeno;


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto prim1 = get_input<PrimitiveObject>("prim1");
        auto prim2 = get_input<PrimitiveObject>("prim2");
        auto op_type = get_param<std::string>("op_type");

        Eigen::MatrixXd VA,VB,VC;
        Eigen::VectorXi J,I;
        Eigen::MatrixXi FA,FB,FC;

        igl::MeshBooleanType boolean_type;

        if (op_type == "Union") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
        } else if (op_type == "Intersect") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
        } else if (op_type == "Minus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
        } else if (op_type == "XOR") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_XOR;
        } else if (op_type == "Resolve") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_RESOLVE;
        }

        igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC,J);

        auto res = std::make_shared<PrimitiveObject>();
        auto &pos = res->add_attr<vec3f>("pos");

        set_output("prim", std::move(res));
    }
};

ZENO_DEFNODE(PrimitiveBooleanOp)({
    {
    "prim1", "prim2"
    },
    {
    "prim",
    },
    {
    {"enum Union Intersect Minus XOR Resolve", "op_type", "union"},
    },
    {"cgmesh"},
});

}
