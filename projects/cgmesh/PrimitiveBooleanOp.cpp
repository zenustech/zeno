#include <zeno/zeno.h>
#include <zeno/utils/vec.h>
#include <zeno/types/PrimitiveObject.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <Eigen/Core>

namespace {

using namespace zeno;


void prim_to_eigen(Eigen::MatrixXd &V, Eigen::MatrixXi &F, PrimitiveObject const *prim) {
    auto &verts = prim->attr<vec3f>("pos");
    for (int i = 0; i < prim->size(); i++) {
        auto pos = verts[i];
        V.push_back({pos[0], pos[1], pos[2]});
    }

    for (int i = 0; i < prim->tris.size(); i++) {
        auto ind = prim->tris[i];
        F.push_back({ind[0], ind[1], ind[2]});
    }
}


void eigen_to_prim(Eigen::MatrixXd const &V, Eigen::MatrixXi const &F, PrimitiveObject *prim) {
    auto &verts = prim->attr<vec3f>("pos");
    prim->verts.clear();
    for (int i = 0; i < V->size(); i++) {
        auto const &pos = V[i];
        verts.emplace_back(V[0], V[1], V[2]);
    }
    prim->resize(verts);

    prim->tris.clear();
    for (int i = 0; i < F.size(); i++) {
        auto const &ind = F[i];
        prim->tris.emplace_back(ind[0], ind[1], ind[2]);
    }
}


struct PrimitiveBooleanOp : INode {
    virtual void apply() override {
        auto primA = get_input<PrimitiveObject>("primA");
        auto primB = get_input<PrimitiveObject>("primB");
        auto op_type = get_param<std::string>("op_type");

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

        Eigen::MatrixXd VA, VB, VC;
        Eigen::MatrixXi FA, FB, FC;
        Eigen::VectorXi J;

        prim_to_eigen(VA, FA, primA.get());
        prim_to_eigen(VB, FB, primB.get());

        igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC,J);

        auto attrName = get_param<std::string>("attrName");
        if (attrName.size()) {
            auto attrValA = get_input<NumericObject>("attrValA")->value;
            auto attrValB = get_input<NumericObject>("attrValB")->value;

            std::visit([&] (auto valA) {
                auto valB = std::get<decltype(valA)>(attrValB);
            }, attrValA);
        }

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
    {"enum Union Intersect Minus XOR Resolve", "op_type", "union"},
    {"string", "attrName", ""},
    },
    {"cgmesh"},
});

}
