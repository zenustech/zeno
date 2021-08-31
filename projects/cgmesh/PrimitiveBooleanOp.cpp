#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <Eigen/Core>

namespace {

using namespace zeno;


auto prim_to_eigen(PrimitiveObject const *prim) {
    Eigen::MatrixXd V(prim->size(), 3);
    Eigen::MatrixXi F(prim->tris.size(), 3);
    auto &verts = prim->attr<vec3f>("pos");
    for (int i = 0; i < prim->size(); i++) {
        auto const &pos = verts[i];
        V.row(i) = Eigen::RowVector3d(pos[0], pos[1], pos[2]);
    }

    for (int i = 0; i < prim->tris.size(); i++) {
        auto const &ind = prim->tris[i];
        F.row(i) = Eigen::RowVector3i(ind[0], ind[1], ind[2]);
    }
    return std::make_pair(V, F);
}


void eigen_to_prim(Eigen::MatrixXd const &V, Eigen::MatrixXi const &F, PrimitiveObject *prim) {
    auto &verts = prim->add_attr<vec3f>("pos");
    verts.clear();
    for (int i = 0; i < V.rows(); i++) {
        auto const &pos = V.row(i);
        verts.emplace_back(V(0), V(1), V(2));
    }
    prim->resize(verts.size());

    prim->tris.clear();
    for (int i = 0; i < F.rows(); i++) {
        auto const &ind = F.row(i);
        prim->tris.emplace_back(ind(0), ind(1), ind(2));
    }
}


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");
        auto op_type = get_param<std::string>("op_type");

        printf("0\n");
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
        } else {
          throw Exception("bad boolean op type: " + op_type);
        }

        printf("1\n");
        auto [VA, FA] = prim_to_eigen(primA.get());
        printf("2\n");
        auto [VB, FB] = prim_to_eigen(primB.get());
        printf("3\n");

        Eigen::MatrixXd VC;
        Eigen::MatrixXi FC;
        Eigen::VectorXi J;
        igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, boolean_type, VC, FC, J);
        printf("4\n");

        /*auto attrName = get_param<std::string>("attrName");
        printf("5\n");
        if (attrName.size()) {
            auto attrValA = get_input<NumericObject>("attrValA")->value;
            auto attrValB = get_input<NumericObject>("attrValB")->value;

            std::visit([&] (auto valA) {
                auto valB = std::get<decltype(valA)>(attrValB);
            }, attrValA);
        }*/

        printf("6\n");
        auto primC = std::make_shared<PrimitiveObject>();
        eigen_to_prim(VC, FC, primC.get());

        printf("7\n");
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
    {"enum Union Intersect Minus XOR Resolve", "op_type", "union"},
    {"string", "attrName", ""},
    },
    {"cgmesh"},
});

}
