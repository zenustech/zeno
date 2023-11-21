// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "./SurfaceCurvature.h"
#include "./SurfaceNormals.h"

#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/color.h"

namespace zeno {
namespace pmp {

SurfaceCurvature::SurfaceCurvature(SurfaceMesh *mesh) : mesh_(mesh),
                                                        min_curv_tag_("curv_min"),
                                                        max_curv_tag_("curv_max"),
                                                        gaussian_curv_tag_("curv_gaussian") {
    vertice_num_ = mesh_->prim->verts.size();
    edge_num_ = mesh_->prim->lines.size();
    mesh_->prim->verts.add_attr<float>(min_curv_tag_);
    mesh_->prim->verts.add_attr<float>(max_curv_tag_);
    mesh_->prim->verts.add_attr<float>(gaussian_curv_tag_);
}

SurfaceCurvature::SurfaceCurvature(SurfaceMesh *mesh,
                                   std::string min_curv_tag,
                                   std::string max_curv_tag,
                                   std::string gaussian_curv_tag)
                                        : mesh_(mesh),
                                          min_curv_tag_(min_curv_tag),
                                          max_curv_tag_(max_curv_tag),
                                          gaussian_curv_tag_(gaussian_curv_tag) {
    vertice_num_ = mesh_->prim->verts.size();
    edge_num_ = mesh_->prim->lines.size();
    mesh_->prim->verts.add_attr<float>(min_curv_tag_);
    mesh_->prim->verts.add_attr<float>(max_curv_tag_);
    mesh_->prim->verts.add_attr<float>(gaussian_curv_tag_);
}

SurfaceCurvature::~SurfaceCurvature() {
    // delete outside
    // mesh_->prim->verts.erase_attr(min_curv_tag);
    // mesh_->prim->verts.erase_attr(max_curv_tag);
    // mesh_->prim->verts.erase_attr(gaussian_curv_tag);
}

void SurfaceCurvature::analyze_tensor(unsigned int post_smoothing_steps) {
    auto &min_curvature = mesh_->prim->verts.attr<float>(min_curv_tag_);
    auto &max_curvature = mesh_->prim->verts.attr<float>(max_curv_tag_);
    auto &gaussian_curvature = mesh_->prim->verts.attr<float>(gaussian_curv_tag_);
    auto area = mesh_->prim->verts.add_attr<float>("curv_area", 0.0);
    auto normal = mesh_->prim->tris.add_attr<vec3f>("curv_normal");
    auto evec = mesh_->prim->lines.add_attr<vec3f>("curv_evec", vec3f(0, 0, 0));
    auto angle = mesh_->prim->lines.add_attr<float>("curv_angle", 0.0);

    vec3f p0, p1, n0, n1, ev;
    float l, A, beta, a1, a2, a3;
    Eigen::Matrix3f tensor;

    float eval1, eval2, eval3;
    float kmin, kmax;
    vec3f evec1, evec2, evec3;

    std::vector<int> neighborhood;
    neighborhood.reserve(15);

    zs::CppTimer timer, timer0;

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    auto &vdeleted = mesh_->prim->verts.attr<int>("v_deleted");
    // precompute Voronoi area per vertex
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        area[v] = mesh_->voronoi_area(v);
    }
#if PMP_ENABLE_PROFILE
    timer.tock("    voronoi are per vertex");
#endif

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    auto &fdeleted = mesh_->prim->tris.attr<int>("f_deleted");
#if 0
    // precompute face normals
    if (!mesh_->has_garbage_)
#pragma omp parallel for
        for (int f = 0; f < mesh_->faces_size_; ++f) {
            if (fdeleted[f])
                continue;
            normal[f] = (vec3f)SurfaceNormals::compute_face_normal(mesh_, f);
        }
#else
    {
        auto &pos = mesh_->prim->attr<vec3f>("pos");
        if (!mesh_->has_garbage_)
#pragma omp parallel for
            for (int v = 0; v < mesh_->vertices_size_; ++v) {
                int h0, h1, h2;
                vec3f p, q, r, pq, qr, pr;
                for (auto h : mesh_->halfedges(v)) {
                    auto f = mesh_->hconn_[h].face_;
                    if (f != PMP_MAX_INDEX && !fdeleted[f]) {
                        h0 = h;
                        h1 = mesh_->next_halfedge(h0);
                        h2 = mesh_->next_halfedge(h1);
                        // three vertex positions
                        auto p = pos[mesh_->to_vertex(h2)];
                        auto q = pos[mesh_->to_vertex(h0)];
                        auto r = pos[mesh_->to_vertex(h1)];

                        // edge vectors
                        (pq = q) -= p;
                        (qr = r) -= q;
                        (pr = r) -= p;

                        normal[f] = normalize(cross(pq, pr));
                        // normal[h] = normalize(cross(p2 - p1, p0 - p1));
                    }
                }
            }
    }
#endif
#if PMP_ENABLE_PROFILE
    timer.tock("    compute_face_normal");
#endif

    auto &pos = mesh_->prim->attr<vec3f>("pos");
    auto &edeleted = mesh_->prim->lines.attr<int>("e_deleted");

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
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
#if PMP_ENABLE_PROFILE
    timer.tock("    compute_dihedral_angle...");
#endif

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif

    double accum = 0.;

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

#if PMP_ENABLE_PROFILE
            timer0.tick();
#endif
            // Eigen-decomposition
            bool ok = symmetric_eigendecomposition(tensor, eval1, eval2, eval3, evec1, evec2, evec3);
#if PMP_ENABLE_PROFILE
            timer0.tock();
            accum += timer0.elapsed();
#endif

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

        min_curvature[v] = kmin;
        max_curvature[v] = kmax;
    }

#if PMP_ENABLE_PROFILE
    timer.tock("    compute curvature tensor for each vertex");
    fmt::print(fg(fmt::color::green), "     symmetric_decomp takes {}\n", accum);
#endif

    // clean-up properties
    mesh_->prim->verts.erase_attr("curv_area");
    mesh_->prim->lines.erase_attr("curv_evec");
    mesh_->prim->lines.erase_attr("curv_angle");
    mesh_->prim->tris.erase_attr("curv_normal");

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    // smooth curvature values
    smooth_curvatures(post_smoothing_steps);
#if PMP_ENABLE_PROFILE
    timer.tock("    smooth curvature values");
#endif

    // calculate gaussian curvature
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        gaussian_curvature[v] = min_curvature[v] * max_curvature[v];
    }
}

void SurfaceCurvature::smooth_curvatures(unsigned int iterations) {
    float kmin, kmax;
    float weight, sum_weights;

    // properties
    auto vfeature = mesh_->prim->verts.attr<int>("v_feature");
    auto cotan = mesh_->prim->lines.add_attr<float>("curv_cotan");
    auto &vdeleted = mesh_->prim->verts.attr<int>("v_deleted");
    auto &edeleted = mesh_->prim->lines.attr<int>("e_deleted");
    auto &min_curvature = mesh_->prim->verts.attr<float>(min_curv_tag_);
    auto &max_curvature = mesh_->prim->verts.attr<float>(max_curv_tag_);

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
                kmin += weight * min_curvature[tv];
                kmax += weight * max_curvature[tv];
            }

            if (sum_weights) {
                min_curvature[v] = kmin / sum_weights;
                max_curvature[v] = kmax / sum_weights;
            }
        }
    }

    // remove property
    mesh_->prim->lines.erase_attr("curv_cotan");
}

} // namespace pmp
} // namespace zeno
