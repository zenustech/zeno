#pragma once

#include <zeno/types/PrimitiveObject.h>
#include <Eigen/Core>


namespace zeno {

inline auto prim_to_eigen(PrimitiveObject const *prim) {
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

// defined in PrimitiveMeshingFix.cpp:
std::pair<Eigen::MatrixXd, Eigen::MatrixXi> prim_to_eigen_with_fix(PrimitiveObject const *primA);

inline void eigen_to_prim(Eigen::MatrixXd const &V, Eigen::MatrixXi const &F, PrimitiveObject *prim) {
    auto &verts = prim->add_attr<vec3f>("pos");
    verts.clear();
    for (int i = 0; i < V.rows(); i++) {
        auto const &pos = V.row(i);
        verts.emplace_back(pos(0), pos(1), pos(2));
    }
    prim->resize(verts.size());

    prim->tris.clear();
    for (int i = 0; i < F.rows(); i++) {
        auto const &ind = F.row(i);
        prim->tris.emplace_back(ind(0), ind(1), ind(2));
    }
}

}
