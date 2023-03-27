// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "./SurfaceCurvature.h"
#include "./SurfaceNormals.h"

namespace zeno {
namespace pmp {

SurfaceCurvature::SurfaceCurvature(SurfaceMesh* mesh) : mesh_(mesh) {
    vertice_num_ = mesh_->prim_->verts.size();
    edge_num_ = mesh_->prim_->lines.size();
    min_curvature_ = mesh_->prim_->verts.add_attr<float>("curv_min");
    max_curvature_ = mesh_->prim_->verts.add_attr<float>("curv_max");
}

SurfaceCurvature::~SurfaceCurvature() {
    mesh_->prim_->verts.erase_attr("curv_min");
    mesh_->prim_->verts.erase_attr("curv_max");
}

void SurfaceCurvature::analyze_tensor(unsigned int post_smoothing_steps) {
    auto area = mesh_->prim_->verts.add_attr<float>("curv_area", 0.0);
    auto normal = mesh_->prim_->tris.add_attr<vec3f>("curv_normal");
    auto evec = mesh_->prim_->lines.add_attr<vec3f>("curv_evec", vec3f(0, 0, 0));
    auto angle = mesh_->prim_->lines.add_attr<float>("curv_angle", 0.0);

    vec3f p0, p1, n0, n1, ev;
    float l, A, beta, a1, a2, a3;
    Eigen::Matrix3f tensor;

    float eval1, eval2, eval3;
    float kmin, kmax;
    vec3f evec1, evec2, evec3;

    std::vector<int> neighborhood;
    neighborhood.reserve(15);

    auto& vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    // precompute Voronoi area per vertex
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        area[v] = mesh_->voronoi_area(v);
    }

    auto& fdeleted = mesh_->prim_->tris.attr<int>("f_deleted");
    // precompute face normals
    for (int f = 0; f < mesh_->faces_size_; ++f) {
        if (mesh_->has_garbage_ && fdeleted[f])
            continue;
        normal[f] = (vec3f)SurfaceNormals::compute_face_normal(mesh_, f);
    }

    auto& pos = mesh_->prim_->attr<vec3f>("pos");
    auto& edeleted = mesh_->prim_->lines.attr<int>("e_deleted");

    // precompute dihedralAngle*edge_length*edge per edge
    for (int e = 0; e < mesh_->lines_size_; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        auto h0 = e << 1;
        auto h1 = e << 1 | 1;
        auto f0 = mesh_->hconn_[h0].face_;
        auto f1 = mesh_->hconn_[h1].face_;
        if (f0 != PMP_MAX_INDEX && f1 != PMP_MAX_INDEX) {
            n0 = normal[f0];
            n1 = normal[f1];
            ev = pos[mesh_->to_vertex(h0)];
            ev -= pos[mesh_->to_vertex(h1)];
            l = length(ev);
            ev /= l;
            l *= 0.5; // only consider half of the edge (matchig Voronoi area)
            angle[e] = atan2(dot(cross(n0, n1), ev), dot(n0, n1));
            evec[e] = sqrt(l) * ev;
        }
    }

    // compute curvature tensor for each vertex
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        kmin = 0.0;
        kmax = 0.0;

        if (!mesh_->is_isolated(v)) {
            // one-ring or two-ring neighborhood?
            neighborhood.clear();
            neighborhood.push_back(v);

            A = 0.0;
            tensor.setZero();

            // compute tensor over vertex neighborhood stored in vertices
            for (auto nit : neighborhood) {
                // accumulate tensor from dihedral angles around vertices
                for (auto hv : mesh_->halfedges(nit)) {
                    auto ee = hv >> 1;
                    ev = evec[ee];
                    beta = angle[ee];
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            tensor(i, j) += beta * ev[i] * ev[j];
                }

                // accumulate area
                A += area[nit];
            }

            // normalize tensor by accumulated
            tensor /= A;

            // Eigen-decomposition
            bool ok = symmetric_eigendecomposition(tensor, eval1, eval2, eval3,
                                                   evec1, evec2, evec3);
            if (ok) {
                // curvature values:
                //   normal vector -> eval with smallest absolute value
                //   evals are sorted in decreasing order
                a1 = fabs(eval1);
                a2 = fabs(eval2);
                a3 = fabs(eval3);
                if (a1 < a2) {
                    if (a1 < a3) {
                        // e1 is normal
                        kmax = eval2;
                        kmin = eval3;
                    } else {
                        // e3 is normal
                        kmax = eval1;
                        kmin = eval2;
                    }
                } else {
                    if (a2 < a3) {
                        // e2 is normal
                        kmax = eval1;
                        kmin = eval3;
                    } else {
                        // e3 is normal
                        kmax = eval1;
                        kmin = eval2;
                    }
                }
            }
        }

        assert(kmin <= kmax);

        min_curvature_[v] = kmin;
        max_curvature_[v] = kmax;
    }

    // clean-up properties
    mesh_->prim_->verts.erase_attr("curv_area");
    mesh_->prim_->lines.erase_attr("curv_evec");
    mesh_->prim_->lines.erase_attr("curv_angle");
    mesh_->prim_->tris.erase_attr("curv_normal");

    // smooth curvature values
    smooth_curvatures(post_smoothing_steps);
}

void SurfaceCurvature::smooth_curvatures(unsigned int iterations) {
    float kmin, kmax;
    float weight, sum_weights;

    // properties
    auto vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto cotan = mesh_->prim_->lines.add_attr<float>("curv_cotan");
    auto& vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto& edeleted = mesh_->prim_->lines.attr<int>("e_deleted");

    // cotan weight per edge
    for (int e = 0; e < mesh_->lines_size_; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        cotan[e] = mesh_->cotan_weight(e);
    }

    for (unsigned int i = 0; i < iterations; ++i) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            // don't smooth feature vertices
            if (vfeature.size() > 0 && vfeature[v] == 1)
                continue;

            kmin = kmax = sum_weights = 0.0;

            for (auto vh : mesh_->halfedges(v)) {
                auto tv = mesh_->to_vertex(vh);

                // don't consider feature vertices (high curvature)
                if (vfeature.size() > 0 && vfeature[tv] == 1)
                    continue;

                weight = std::max(0.0f, mesh_->cotan_weight(vh >> 1));
                sum_weights += weight;
                kmin += weight * min_curvature_[tv];
                kmax += weight * max_curvature_[tv];
            }

            if (sum_weights) {
                min_curvature_[v] = kmin / sum_weights;
                max_curvature_[v] = kmax / sum_weights;
            }
        }
    }

    // remove property
    mesh_->prim_->lines.erase_attr("curv_cotan");
}

} // namespace pmp
} // namespace zeno
