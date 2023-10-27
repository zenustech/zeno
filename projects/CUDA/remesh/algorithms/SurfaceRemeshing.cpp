// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "./SurfaceRemeshing.h"
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "./SurfaceCurvature.h"
#include "./SurfaceNormals.h"
#include "./TriangleKdTree.h"

#include "zensim/profile/CppTimers.hpp"
#include "zensim/zpc_tpls/fmt/core.h"

namespace zeno {
namespace pmp {

const vec3f barycentric_coordinates(const vec3f &p, const vec3f &u, const vec3f &v, const vec3f &w) {
    vec3f result(1.0 / 3.0); // default: barycenter

    vec3f vu = v - u, wu = w - u, pu = p - u;

    // find largest absolute coodinate of normal
    float nx = vu[1] * wu[2] - vu[2] * wu[1], ny = vu[2] * wu[0] - vu[0] * wu[2], nz = vu[0] * wu[1] - vu[1] * wu[0],
          ax = fabs(nx), ay = fabs(ny), az = fabs(nz);

    unsigned char maxCoord;

    if (ax > ay) {
        if (ax > az) {
            maxCoord = 0;
        } else {
            maxCoord = 2;
        }
    } else {
        if (ay > az) {
            maxCoord = 1;
        } else {
            maxCoord = 2;
        }
    }

    // solve 2D problem
    switch (maxCoord) {
    case 0: {
        if (1.0 + ax != 1.0) {
            result[1] = 1.0 + (pu[1] * wu[2] - pu[2] * wu[1]) / nx - 1.0;
            result[2] = 1.0 + (vu[1] * pu[2] - vu[2] * pu[1]) / nx - 1.0;
            result[0] = 1.0 - result[1] - result[2];
        }
        break;
    }

    case 1: {
        if (1.0 + ay != 1.0) {
            result[1] = 1.0 + (pu[2] * wu[0] - pu[0] * wu[2]) / ny - 1.0;
            result[2] = 1.0 + (vu[2] * pu[0] - vu[0] * pu[2]) / ny - 1.0;
            result[0] = 1.0 - result[1] - result[2];
        }
        break;
    }

    case 2: {
        if (1.0 + az != 1.0) {
            result[1] = 1.0 + (pu[0] * wu[1] - pu[1] * wu[0]) / nz - 1.0;
            result[2] = 1.0 + (vu[0] * pu[1] - vu[1] * pu[0]) / nz - 1.0;
            result[0] = 1.0 - result[1] - result[2];
        }
        break;
    }
    }

    return result;
}

SurfaceRemeshing::SurfaceRemeshing(SurfaceMesh *mesh, std::string line_pick_tag)
    : mesh_(mesh), line_pick_tag_(line_pick_tag), refmesh_(nullptr), kd_tree_(nullptr) {

    auto vnormal = mesh_->prim_->verts.add_attr<vec3f>("v_normal");

    SurfaceNormals::compute_vertex_normals(mesh);
}

SurfaceRemeshing::~SurfaceRemeshing() = default;

void SurfaceRemeshing::uniform_remeshing(float edge_length, unsigned int iterations, bool use_projection) {
    if (mesh_->prim_->quads.size() > 0 || mesh_->prim_->polys.size() > 0) {
        zeno::log_error("Not a triangle mesh!");
        return;
    }

    uniform_ = true;
    use_projection_ = use_projection;
    target_edge_length_ = edge_length;

    preprocessing();

    for (unsigned int i = 0; i < iterations; ++i) {
        if (split_long_edges() == PMP_MAX_INDEX) {
            return;
        }
        SurfaceNormals::compute_vertex_normals(mesh_);
        collapse_short_edges();
        collapse_crosses();
        flip_edges();
        laplacian_smoothing();
    }

    remove_caps();
    postprocessing();
    check_triangles();
}

void SurfaceRemeshing::adaptive_remeshing(float min_edge_length, float max_edge_length, float approx_error,
                                          unsigned int iterations, bool use_projection) {
    if (mesh_->prim_->quads.size() > 0 || mesh_->prim_->polys.size() > 0) {
        zeno::log_error("Not a triangle mesh!");
        return;
    }

    uniform_ = false;
    min_edge_length_ = min_edge_length;
    max_edge_length_ = max_edge_length;
    approx_error_ = approx_error;
    use_projection_ = use_projection;

    zs::CppTimer timer;

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    preprocessing();
#if PMP_ENABLE_PROFILE
    timer.tock("preprocessing");
#endif

    for (unsigned int i = 0; i < iterations; ++i) {
        if (split_long_edges() == PMP_MAX_INDEX) {
            return;
        }
#if PMP_ENABLE_PROFILE
        fmt::print("start iteration [{}]\n", i);
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        SurfaceNormals::compute_vertex_normals(mesh_);
#if PMP_ENABLE_PROFILE
        timer.tock("compute_vertex_normals");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        collapse_short_edges();
        collapse_crosses();
#if PMP_ENABLE_PROFILE
        timer.tock("collapse_short_edges");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        flip_edges();
#if PMP_ENABLE_PROFILE
        timer.tock("flip_edges");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        laplacian_smoothing();
#if PMP_ENABLE_PROFILE
        timer.tock("laplacian_smoothing");
#endif

    }

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    remove_caps();
#if PMP_ENABLE_PROFILE
    timer.tock("remove_caps");
#endif

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    postprocessing();
#if PMP_ENABLE_PROFILE
    timer.tock("postprocessing");
#endif
#if PMP_ENABLE_PROFILE
    timer.tick();
#endif
    check_triangles();
#if PMP_ENABLE_PROFILE
    timer.tock("check_triangles");
#endif
}

void SurfaceRemeshing::check_triangles() {
    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &fdeleted = mesh_->prim_->tris.attr<int>("f_deleted");
    bool degenerated = false;

    for (int t = 0; t < mesh_->faces_size_; ++t) {
        if (mesh_->has_garbage_ && fdeleted[t])
            continue;
        auto tri = mesh_->prim_->tris[t];
        auto area = length(cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]]));
        if (area < std::numeric_limits<float>::epsilon()) {
            degenerated = true;
        }
    }
    if (degenerated) {
        zeno::log_warn("remesh: Degenerated triangle detected!");
    }
}

void SurfaceRemeshing::preprocessing() {
    // properties
    auto &vfeature = mesh_->prim_->verts.add_attr<int>("v_feature", 0);
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &vduplicate = mesh_->prim_->verts.attr<int>("v_duplicate");
    auto &vsizing = mesh_->prim_->verts.add_attr<float>("v_sizing");
    auto &vlocked = mesh_->prim_->verts.add_attr<int>("v_locked", 0);
    auto &elocked = mesh_->prim_->lines.add_attr<int>("e_locked", 0);
    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");

    // feature vertices
    for (int e = 0; e < mesh_->lines_size_; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (efeature[e] == 1) {
            vfeature[mesh_->prim_->lines[e][0]] = 1;
            vfeature[mesh_->prim_->lines[e][1]] = 1;
        }
    }

    zs::CppTimer timer;

#if PMP_ENABLE_PROFILE
    timer.tick();
#endif

    // lock feature corners
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        if (vduplicate[v] != v) {
            vlocked[v] = vlocked[vduplicate[v]] = 1;
        }
        if (vfeature[v]) {
            int c = 0;
            for (auto h : mesh_->halfedges(v)) {
                if (efeature[h >> 1]) {
                    ++c;
                }
                if (mesh_->hconn_[h].face_ == PMP_MAX_INDEX) {
                    vlocked[v] = 1;
                }
            }

            if (c != 2) {
                vlocked[v] = 1;
            }
        }
    }

#if PMP_ENABLE_PROFILE
    timer.tock("lock feature corners");
#endif

    // compute sizing field
    if (uniform_) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            vsizing[v] = target_edge_length_;
        }
    } else {

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif

        // compute curvature for all mesh vertices, using cotan or Cohen-Steiner
        // don't use two-ring neighborhood, since we otherwise compute
        // curvature over sharp features edges, leading to high curvatures.
        // prefer tensor analysis over cotan-Laplace, since the former is more
        // robust and gives better results on the boundary.
        SurfaceCurvature curv(mesh_);
        curv.analyze_tensor(1);

#if PMP_ENABLE_PROFILE
        timer.tock("analyze_tensor");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        // use vsizing_ to store/smooth curvatures to avoid another vertex property

        // curvature values for feature vertices and boundary vertices
        // are not meaningful. mark them as negative values.
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (mesh_->is_boundary_v(v) || (vfeature.size() > 0 && vfeature[v] == 1))
                vsizing[v] = -1.0;
            else
                vsizing[v] = curv.max_abs_curvature(v);
        }

#if PMP_ENABLE_PROFILE
        timer.tock("compute curvature values");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif

        // curvature values might be noisy. smooth them.
        // don't consider feature vertices' curvatures.
        // don't consider boundary vertices' curvatures.
        // do this for two iterations, to propagate curvatures
        // from non-feature regions to feature vertices.
        for (int iters = 0; iters < 2; ++iters) {
            for (int v = 0; v < mesh_->vertices_size_; ++v) {
                if (mesh_->has_garbage_ && vdeleted[v])
                    continue;
                float w, ww = 0.0;
                float c, cc = 0.0;

                for (auto h : mesh_->halfedges(v)) {
                    c = vsizing[mesh_->hconn_[h].vertex_];
                    if (c > 0.0) {
                        w = std::max(0.0f, mesh_->cotan_weight(h >> 1));
                        ww += w;
                        cc += w * c;
                    }
                }

                if (ww)
                    cc /= ww;
                vsizing[v] = cc;
            }
        }

#if PMP_ENABLE_PROFILE
        timer.tock("smooth curvature values");
#endif

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif

        // now convert per-vertex curvature into target edge length
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            float c = vsizing[v];

            // get edge length from curvature
            const float r = 1.0 / c;
            const float e = approx_error_;
            float h;
            if (e < r) {
                // see mathworld: "circle segment" and "equilateral triangle"
                //h = sqrt(2.0*r*e-e*e) * 3.0 / sqrt(3.0);
                h = sqrt(6.0 * e * r - 3.0 * e * e); // simplified...
            } else {
                // this does not really make sense
                h = e * 3.0 / sqrt(3.0);
            }

            // clamp to min. and max. edge length
            if (h < min_edge_length_)
                h = min_edge_length_;
            else if (h > max_edge_length_)
                h = max_edge_length_;

            // store target edge length
            vsizing[v] = h;
        }
#if PMP_ENABLE_PROFILE
        timer.tock("convert per-vertex curvature into target edge length");
#endif
    }

    if (use_projection_) {
        // build reference mesh
        refmesh_ = new SurfaceMesh(*mesh_);
        SurfaceNormals::compute_vertex_normals(refmesh_);

        refpoints_ = refmesh_->prim_->attr<vec3f>("pos");
        refnormals_ = refmesh_->prim_->verts.add_attr<vec3f>("v_normal");
        auto &refvdeleted = refmesh_->prim_->verts.attr<int>("v_deleted");

        // copy sizing field from prim_
        refsizing_ = refmesh_->prim_->verts.add_attr<float>("v_sizing");
        for (int v = 0; v < refmesh_->vertices_size_; ++v) {
            if (refmesh_->has_garbage_ && refvdeleted[v])
                continue;
            refsizing_[v] = vsizing[v];
        }

#if PMP_ENABLE_PROFILE
        timer.tick();
#endif
        // build kd-tree
        kd_tree_ = new TriangleKdTree(refmesh_, 0);
#if PMP_ENABLE_PROFILE
        timer.tock("build kd-tree");
#endif
    }
}

void SurfaceRemeshing::postprocessing() {
    // delete kd-tree and reference mesh
    if (use_projection_) {
        delete kd_tree_;
        delete refmesh_;
    }

    // remove properties
    mesh_->prim_->verts.erase_attr("v_feature");
    mesh_->prim_->verts.erase_attr("v_locked");
    mesh_->prim_->lines.erase_attr("e_locked");
    mesh_->prim_->verts.erase_attr("v_sizing");
}

void SurfaceRemeshing::project_to_reference(int v) {
    if (!use_projection_) {
        return;
    }

    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &vnormal = mesh_->prim_->verts.attr<vec3f>("v_normal");
    auto &vsizing = mesh_->prim_->verts.attr<float>("v_sizing");

    // find closest triangle of reference mesh
    TriangleKdTree::NearestNeighbor nn = kd_tree_->nearest(points[v]);
    const vec3f p = nn.nearest;
    const int f = nn.face;

    // get face data
    auto fvIt = refmesh_->prim_->tris[f];

    const vec3f p0 = refpoints_[fvIt[0]];
    const vec3f n0 = refnormals_[fvIt[0]];
    const float s0 = refsizing_[fvIt[0]];

    const vec3f p1 = refpoints_[fvIt[1]];
    const vec3f n1 = refnormals_[fvIt[1]];
    const float s1 = refsizing_[fvIt[1]];

    const vec3f p2 = refpoints_[fvIt[2]];
    const vec3f n2 = refnormals_[fvIt[2]];
    const float s2 = refsizing_[fvIt[2]];

    // get barycentric coordinates
    vec3f b = barycentric_coordinates(p, p0, p1, p2);

    // interpolate normal
    vec3f n;
    n = (n0 * b[0]);
    n += (n1 * b[1]);
    n += (n2 * b[2]);
    n = normalize(n);
    assert(!std::isnan(n[0]));

    // interpolate sizing field
    float s;
    s = (s0 * b[0]);
    s += (s1 * b[1]);
    s += (s2 * b[2]);

    // set result
    points[v] = p;
    vnormal[v] = n;
    vsizing[v] = s;
}

int SurfaceRemeshing::split_long_edges() {
    int vnew, v0, v1;
    int enew, e0, e1;
    int f0, f1, f2, f3;
    bool is_feature, is_boundary;
    int i;

    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &vnormal = mesh_->prim_->verts.attr<vec3f>("v_normal");
    auto &vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &elocked = mesh_->prim_->lines.attr<int>("e_locked");
    auto &vsizing = mesh_->prim_->verts.attr<float>("v_sizing");
    auto &vduplicate = mesh_->prim_->verts.attr<int>("v_duplicate");

    int lines = mesh_->lines_size_;
    for (int e = 0; e < lines; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        v0 = mesh_->prim_->lines[e][0];
        v1 = mesh_->prim_->lines[e][1];

        if (!elocked[e] && is_too_long(v0, v1)) {
            const vec3f &p0 = points[v0];
            const vec3f &p1 = points[v1];

            is_feature = efeature[e];
            is_boundary = mesh_->is_boundary_e(e);

            vnew = mesh_->new_vertex((p0 + p1) * 0.5f);
            if (vnew == PMP_MAX_INDEX) {
                return PMP_MAX_INDEX;
            }

            int new_lines, new_faces;
            if (mesh_->split(e, vnew, new_lines, new_faces) == PMP_MAX_INDEX) {
                return PMP_MAX_INDEX;
            }
            for (int ii = 1; ii <= new_lines; ++ii) {
                mesh_->prim_->lines.foreach_attr<zeno::AttrAcceptAll>([&](auto const &key, auto &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        arr.push_back(zeno::vec3f(0, 0, 0));
                    } else if constexpr (std::is_same_v<T, float>) {
                        arr.push_back(0);
                    } else if constexpr (std::is_same_v<T, zeno::vec3i>) {
                        arr.push_back(zeno::vec3i(0, 0, 0));
                    } else if constexpr (std::is_same_v<T, int>) {
                        arr.push_back(0);
                    } else if constexpr (std::is_same_v<T, zeno::vec2f>) {
                        arr.push_back(zeno::vec2f(0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec2i>) {
                        arr.push_back(zeno::vec2i(0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec4f>) {
                        arr.push_back(zeno::vec4f(0, 0, 0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec4i>) {
                        arr.push_back(zeno::vec4i(0, 0, 0, 0));
                    }
                });
            }
            for (int ii = 1; ii <= new_faces; ++ii) {
                mesh_->prim_->tris.foreach_attr<zeno::AttrAcceptAll>([&](auto const &key, auto &arr) {
                    using T = std::decay_t<decltype(arr[0])>;
                    if constexpr (std::is_same_v<T, zeno::vec3f>) {
                        arr.push_back(zeno::vec3f(0, 0, 0));
                    } else if constexpr (std::is_same_v<T, float>) {
                        arr.push_back(0);
                    } else if constexpr (std::is_same_v<T, zeno::vec3i>) {
                        arr.push_back(zeno::vec3i(0, 0, 0));
                    } else if constexpr (std::is_same_v<T, int>) {
                        arr.push_back(0);
                    } else if constexpr (std::is_same_v<T, zeno::vec2f>) {
                        arr.push_back(zeno::vec2f(0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec2i>) {
                        arr.push_back(zeno::vec2i(0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec4f>) {
                        arr.push_back(zeno::vec4f(0, 0, 0, 0));
                    } else if constexpr (std::is_same_v<T, zeno::vec4i>) {
                        arr.push_back(zeno::vec4i(0, 0, 0, 0));
                    }
                });
            }

            // need normal or sizing for adaptive refinement
            mesh_->prim_->verts.foreach_attr<zeno::AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (std::is_same_v<T, zeno::vec3f>) {
                    if (key == "v_normal") {
                        arr.push_back(SurfaceNormals::compute_vertex_normal(mesh_, vnew));
                    } else {
                        arr.push_back(0.5f * (arr[v0] + arr[v1]));
                    }
                } else if constexpr (std::is_same_v<T, float>) {
                    if (key == "v_sizing") {
                        arr.push_back(0.5f * (vsizing[v0] + vsizing[v1]));
                    } else {
                        arr.push_back(0.5f * (arr[v0] + arr[v1]));
                    }
                } else if constexpr (std::is_same_v<T, zeno::vec3i>) {
                    arr.push_back(zeno::vec3i(0, 0, 0));
                } else if constexpr (std::is_same_v<T, int>) {
                    if (key == "v_duplicate") {
                        arr.push_back(vnew);
                    } else if (key == "v_locked" || key == "v_feature" || key == "v_deleted") {
                        arr.push_back(0);
                    } else {
                        arr.push_back(0.5f * (arr[v0] + arr[v1]));
                    }
                } else if constexpr (std::is_same_v<T, zeno::vec2f>) {
                    arr.push_back(0.5f * (arr[v0] + arr[v1]));
                } else if constexpr (std::is_same_v<T, zeno::vec2i>) {
                    arr.push_back(0.5f * (arr[v0] + arr[v1]));
                } else if constexpr (std::is_same_v<T, zeno::vec4f>) {
                    arr.push_back(0.5f * (arr[v0] + arr[v1]));
                } else if constexpr (std::is_same_v<T, zeno::vec4i>) {
                    arr.push_back(0.5f * (arr[v0] + arr[v1]));
                }
            });

            if (is_feature) {
                enew = is_boundary ? mesh_->lines_size_ - 2 : mesh_->lines_size_ - 3;
                efeature[enew] = 1;
                vfeature[vnew] = 1;
            } else {
                project_to_reference(vnew);
            }
        }
    }
    return 0;
}

void SurfaceRemeshing::collapse_short_edges() {
    int v0, v1;
    int h0, h1, h01, h10;
    bool b0, b1, l0, l1, f0, f1;
    int i;
    bool hcol01, hcol10;

    auto &vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &elocked = mesh_->prim_->lines.attr<int>("e_locked");

    int lines = mesh_->lines_size_;
    for (int e = 0; e < lines; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (!edeleted[e] && !elocked[e]) {
            h10 = e << 1;
            h01 = e << 1 | 1;
            v0 = mesh_->to_vertex(h10);
            v1 = mesh_->to_vertex(h01);
            if (is_too_short(v0, v1)) {
                // get status
                b0 = mesh_->is_boundary_v(v0);
                b1 = mesh_->is_boundary_v(v1);
                l0 = vlocked[v0];
                l1 = vlocked[v1];
                f0 = vfeature[v0];
                f1 = vfeature[v1];
                hcol01 = hcol10 = true;

                // boundary rules
                if (b0 && b1) {
                    if (!mesh_->is_boundary_e(e))
                        continue;
                } else if (b0)
                    hcol01 = false;
                else if (b1)
                    hcol10 = false;

                // locked rules
                if (l0 && l1)
                    continue;
                else if (l0)
                    hcol01 = false;
                else if (l1)
                    hcol10 = false;

                // feature rules
                if (f0 && f1) {
                    // edge must be feature
                    if (!efeature[e])
                        continue;

                    // the other two edges removed by collapse must not be features
                    h0 = mesh_->prev_halfedge(h01);
                    h1 = mesh_->next_halfedge(h10);
                    if (efeature[h0 >> 1] || efeature[h1 >> 1])
                        hcol01 = false;
                    // the other two edges removed by collapse must not be features
                    h0 = mesh_->prev_halfedge(h10);
                    h1 = mesh_->next_halfedge(h01);
                    if (efeature[h0 >> 1] || efeature[h1 >> 1])
                        hcol10 = false;
                } else if (f0)
                    hcol01 = false;
                else if (f1)
                    hcol10 = false;

                // topological rules
                mesh_->is_collapse_ok(h01, hcol01, hcol10);

                // both collapses possible: collapse into vertex with higher valence
                if (hcol01 && hcol10) {
                    if (mesh_->valence(v0) < mesh_->valence(v1))
                        hcol10 = false;
                    else
                        hcol01 = false;
                }

                // try v1 -> v0
                if (hcol10) {
                    // don't create too long edges
                    for (auto vv : mesh_->vertices(v1)) {
                        if (is_too_long(v0, vv)) {
                            hcol10 = false;
                            break;
                        }
                    }

                    if (hcol10) {
                        mesh_->collapse(h10);
                    }
                }

                // try v0 -> v1
                else if (hcol01) {
                    // don't create too long edges
                    for (auto vv : mesh_->vertices(v0)) {
                        if (is_too_long(v1, vv)) {
                            hcol01 = false;
                            break;
                        }
                    }

                    if (hcol01) {
                        mesh_->collapse(h01);
                    }
                }
            }
        }
    }
}

void SurfaceRemeshing::collapse_crosses() {
    int v0, v1;
    int h0, h1, h01, h10;
    bool b0, b1, l0, l1, f0, f1;
    int i;
    bool hcol01, hcol10;

    auto &vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &elocked = mesh_->prim_->lines.attr<int>("e_locked");

    int lines = mesh_->lines_size_;
    for (int e = 0; e < lines; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (!edeleted[e] && !elocked[e]) {
            h10 = e << 1;
            h01 = e << 1 | 1;
            v0 = mesh_->to_vertex(h10);
            v1 = mesh_->to_vertex(h01);

            if (is_crosses(v0, v1)) {
                // get status
                b0 = mesh_->is_boundary_v(v0);
                b1 = mesh_->is_boundary_v(v1);
                l0 = vlocked[v0];
                l1 = vlocked[v1];
                f0 = vfeature[v0];
                f1 = vfeature[v1];
                hcol01 = hcol10 = true;

                // boundary rules
                if (b0 && b1) {
                    if (!mesh_->is_boundary_e(e))
                        continue;
                } else if (b0)
                    hcol01 = false;
                else if (b1)
                    hcol10 = false;

                // locked rules
                if (l0 && l1)
                    continue;
                else if (l0)
                    hcol01 = false;
                else if (l1)
                    hcol10 = false;

                // feature rules
                if (f0 && f1) {
                    // edge must be feature
                    if (!efeature[e])
                        continue;

                    // the other two edges removed by collapse must not be features
                    h0 = mesh_->prev_halfedge(h01);
                    h1 = mesh_->next_halfedge(h10);
                    if (efeature[h0 >> 1] || efeature[h1 >> 1])
                        hcol01 = false;
                    // the other two edges removed by collapse must not be features
                    h0 = mesh_->prev_halfedge(h10);
                    h1 = mesh_->next_halfedge(h01);
                    if (efeature[h0 >> 1] || efeature[h1 >> 1])
                        hcol10 = false;
                } else if (f0)
                    hcol01 = false;
                else if (f1)
                    hcol10 = false;

                // topological rules
                mesh_->is_collapse_ok(h01, hcol01, hcol10);

                // both collapses possible: collapse into vertex with higher valence
                if (hcol01 && hcol10) {
                    if (mesh_->valence(v0) < mesh_->valence(v1))
                        hcol10 = false;
                    else
                        hcol01 = false;
                }

                // try v1 -> v0
                if (hcol10) {
                    // don't create too long edges
                    for (auto vv : mesh_->vertices(v1)) {
                        if (is_too_long(v0, vv)) {
                            hcol10 = false;
                            break;
                        }
                    }

                    if (hcol10) {
                        mesh_->collapse(h10);
                    }
                }

                // try v0 -> v1
                else if (hcol01) {
                    // don't create too long edges
                    for (auto vv : mesh_->vertices(v0)) {
                        if (is_too_long(v1, vv)) {
                            hcol01 = false;
                            break;
                        }
                    }

                    if (hcol01) {
                        mesh_->collapse(h01);
                    }
                }
            }
        }
    }
    mesh_->garbage_collection();
}

void SurfaceRemeshing::flip_edges() {
    int v0, v1, v2, v3;
    int h;
    int val0, val1, val2, val3;
    int val_opt0, val_opt1, val_opt2, val_opt3;
    int ve0, ve1, ve2, ve3, ve_before, ve_after;
    int i;

    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &elocked = mesh_->prim_->lines.attr<int>("e_locked");
    // precompute valences
    auto valence = mesh_->prim_->verts.add_attr<int>("valence");
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if (mesh_->has_garbage_ && vdeleted[v])
            continue;
        valence[v] = mesh_->valence(v);
    }

    int lines = mesh_->lines_size_;
    for (int e = 0; e < lines; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (!elocked[e] && !efeature[e]) {
            h = e << 1;
            v0 = mesh_->to_vertex(h);
            v2 = mesh_->to_vertex(mesh_->next_halfedge(h));
            h = e << 1 | 1;
            v1 = mesh_->to_vertex(h);
            v3 = mesh_->to_vertex(mesh_->next_halfedge(h));

            if (!vlocked[v0] && !vlocked[v1] && !vlocked[v2] && !vlocked[v3]) {
                val0 = valence[v0];
                val1 = valence[v1];
                val2 = valence[v2];
                val3 = valence[v3];

                val_opt0 = (mesh_->is_boundary_v(v0) ? 4 : 6);
                val_opt1 = (mesh_->is_boundary_v(v1) ? 4 : 6);
                val_opt2 = (mesh_->is_boundary_v(v2) ? 4 : 6);
                val_opt3 = (mesh_->is_boundary_v(v3) ? 4 : 6);

                ve0 = (val0 - val_opt0);
                ve1 = (val1 - val_opt1);
                ve2 = (val2 - val_opt2);
                ve3 = (val3 - val_opt3);

                ve0 *= ve0;
                ve1 *= ve1;
                ve2 *= ve2;
                ve3 *= ve3;

                ve_before = ve0 + ve1 + ve2 + ve3;

                --val0;
                --val1;
                ++val2;
                ++val3;

                ve0 = (val0 - val_opt0);
                ve1 = (val1 - val_opt1);
                ve2 = (val2 - val_opt2);
                ve3 = (val3 - val_opt3);

                ve0 *= ve0;
                ve1 *= ve1;
                ve2 *= ve2;
                ve3 *= ve3;

                ve_after = ve0 + ve1 + ve2 + ve3;
                if (ve_before > ve_after && mesh_->is_flip_ok(e)) {
                    mesh_->flip(e);
                    --valence[v0];
                    --valence[v1];
                    ++valence[v2];
                    ++valence[v3];
                }
            }
        }
    }

    mesh_->prim_->verts.erase_attr("valence");
}

void SurfaceRemeshing::tangential_smoothing(unsigned int iterations) {
    int v1, v2, v3, vv;
    int e;
    float w, ww;
    vec3f u, n, t, b;

    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &vnormal = mesh_->prim_->verts.attr<vec3f>("v_normal");
    auto &vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &vsizing = mesh_->prim_->verts.attr<float>("v_sizing");
    // add property
    auto update = mesh_->prim_->verts.add_attr<vec3f>("update", vec3f(0.0f));

    // project at the beginning to get valid sizing values and normal vectors
    // for vertices introduced by splitting
    if (use_projection_) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (!mesh_->is_boundary_v(v) && !vlocked[v]) {
                project_to_reference(v);
            }
        }
    }

    for (unsigned int iters = 0; iters < iterations; ++iters) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (!mesh_->is_boundary_v(v) && !vlocked[v]) {
                if (vfeature[v]) {
                    u = vec3f(0.0f);
                    t = vec3f(0.0f);
                    ww = 0;
                    int c = 0;
                    for (auto h : mesh_->halfedges(v)) {
                        if (efeature[h >> 1]) {
                            vv = mesh_->to_vertex(h);

                            b = points[v];
                            b += points[vv];
                            b *= 0.5;

                            w = distance(points[v], points[vv]) / (0.5 * (vsizing[v] + vsizing[vv]));
                            ww += w;
                            u += w * b;

                            if (c == 0) {
                                t += normalize(points[vv] - points[v]);
                                ++c;
                            } else {
                                ++c;
                                t -= normalize(points[vv] - points[v]);
                            }
                        }
                    }

                    assert(c == 2);

                    u *= (1.0 / ww);
                    u -= points[v];
                    t = normalize(t);
                    u = t * dot(u, t);

                    update[v] = u;
                } else {
                    vec3f p(0.0f);
                    bool flag;
                    p = weighted_centroid(v);
                    u = p - points[v];

                    n = vnormal[v];
                    u -= n * dot(u, n);

                    update[v] = u;
                }
            }
        }

        // update vertex positions
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (!mesh_->is_boundary_v(v) && !vlocked[v]) {
                points[v] += update[v];
            }
        }

        // update normal vectors (if not done so through projection)
        SurfaceNormals::compute_vertex_normals(mesh_);
    }

    // project at the end
    if (use_projection_) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (!mesh_->is_boundary_v(v) && !vlocked[v])
                project_to_reference(v);
        }
    }

    // remove property
    mesh_->prim_->verts.erase_attr("update");
}

void SurfaceRemeshing::laplacian_smoothing() {
    int v1, v2, v3, vv;
    int e;
    float w, ww;
    vec3f u, n, t, b;

    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");

    planar_laplacian();
    SurfaceNormals::compute_vertex_normals(mesh_);

    // project at the end
    if (use_projection_) {
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            if (!mesh_->is_boundary_v(v) && !vlocked[v])
                project_to_reference(v);
        }
    }
}

void SurfaceRemeshing::remove_caps() {
    int h;
    int v, vb, vd;
    int fb, fd;
    float a0, a1, amin, aa(::cos(170.0 * M_PI / 180.0));
    vec3f a, b, c, d;

    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &vfeature = mesh_->prim_->verts.attr<int>("v_feature");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &elocked = mesh_->prim_->lines.attr<int>("e_locked");
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");

    for (int e = 0; e < mesh_->lines_size_; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (!elocked[e] && mesh_->is_flip_ok(e)) {
            h = e << 1;
            a = points[mesh_->to_vertex(h)];

            h = mesh_->hconn_[h].next_halfedge_;
            b = points[vb = mesh_->to_vertex(h)];

            h = e << 1 | 1;
            c = points[mesh_->to_vertex(h)];

            h = mesh_->hconn_[h].next_halfedge_;
            d = points[vd = mesh_->to_vertex(h)];

            a0 = dot(normalize(a - b), normalize(c - b));
            a1 = dot(normalize(a - d), normalize(c - d));

            if (a0 < a1) {
                amin = a0;
                v = vb;
            } else {
                amin = a1;
                v = vd;
            }

            // is it a cap?
            if (amin < aa) {
                // feature edge and feature vertex -> seems to be intended
                if (efeature[e] && vfeature[v])
                    continue;

                // project v onto feature edge
                if (efeature[e]) {
                    points[v] = (a + c) * 0.5f;
                }

                // flip
                mesh_->flip(e);
            }
        }
    }
}

vec3f SurfaceRemeshing::minimize_squared_areas(int v, bool &inversable) {

    Eigen::Matrix3f A;
    Eigen::Vector3f b, x;
    A.setZero();
    b.setZero();
    x.setZero();

    auto &points = mesh_->prim_->attr<vec3f>("pos");

    for (int h : mesh_->halfedges(v)) {
        assert(!mesh_->is_boundary(h));

        // get edge opposite to vertex v
        int v0 = mesh_->to_vertex(h);
        int v1 = mesh_->to_vertex(mesh_->next_halfedge(h));
        Eigen::Vector3f p(points[v0][0], points[v0][1], points[v0][2]);
        Eigen::Vector3f q(points[v1][0], points[v1][1], points[v1][2]);
        Eigen::Vector3f d = q - p;
        float w = 1.0 / d.norm();

        // build squared cross-product-with-d matrix
        Eigen::Matrix3f D;
        D(0, 0) = d[1] * d[1] + d[2] * d[2];
        D(1, 1) = d[0] * d[0] + d[2] * d[2];
        D(2, 2) = d[0] * d[0] + d[1] * d[1];
        D(1, 0) = D(0, 1) = -d[0] * d[1];
        D(2, 0) = D(0, 2) = -d[0] * d[2];
        D(1, 2) = D(2, 1) = -d[1] * d[2];
        A += w * D;

        // build right-hand side
        b += w * D * p;
    }

    // compute minimizer
    float det = A.determinant();
    if (fabs(det) < std::numeric_limits<float>::epsilon() * 100 || std::isnan(det)) {
        inversable = false;
    } else {
        inversable = true;
        x = A.inverse() * b;
    }

    vec3f ret(x[0], x[1], x[2]);
    return ret;
}

vec3f SurfaceRemeshing::weighted_centroid(int v) {
    vec3f p(0.0f);
    double ww = 0;

    auto &points = mesh_->prim_->attr<vec3f>("pos");
    auto &vsizing = mesh_->prim_->verts.attr<float>("v_sizing");

    for (int h : mesh_->halfedges(v)) {
        int v1 = v;
        int v2 = mesh_->to_vertex(h);
        int v3 = mesh_->to_vertex(mesh_->next_halfedge(h));

        vec3f b = points[v1];
        b += points[v2];
        b += points[v3];
        b *= (1.0 / 3.0);

        double area = length(cross(points[v2] - points[v1], points[v3] - points[v1]));

        // take care of degenerate faces to avoid all zero weights and division
        // by zero later on
        if (area == 0)
            area = 1.0;

        double w = area / pow((vsizing[v1] + vsizing[v2] + vsizing[v3]) / 3.0, 2.0);

        p += w * b;
        ww += w;
    }

    p /= ww;

    return p;
}

template<class T>
void SurfaceRemeshing::accumulate_laplacian(std::vector<T>& arr, bool calculate_lpzcnt, bool only_feature) {
    auto &vlpzsum = mesh_->prim_->verts.attr<T>("v_lpzsum");
    auto &vlpzcnt = mesh_->prim_->verts.attr<float>("v_lpzcnt");
    auto &efeature = mesh_->prim_->lines.attr<int>(line_pick_tag_);
    auto &edeleted = mesh_->prim_->lines.attr<int>("e_deleted");

    if (!only_feature) {
        for (int e = 0; e < mesh_->lines_size_; ++e) {
            if (mesh_->has_garbage_ && edeleted[e])
                continue;
            if (!mesh_->is_boundary_e(e)) {
                int v0 = mesh_->to_vertex(e << 1);
                int v1 = mesh_->to_vertex(e << 1 | 1);
                vlpzsum[v0] += arr[v1];
                vlpzsum[v1] += arr[v0];
                if (calculate_lpzcnt) {
                    vlpzcnt[v0] += 1.0f;
                    vlpzcnt[v1] += 1.0f;
                }
            }
        }
        // reset border and feature vertices
        for (int e = 0; e < mesh_->lines_size_; ++e) {
            if (mesh_->has_garbage_ && edeleted[e])
                continue;
            if (mesh_->is_boundary_e(e) || efeature[e]) {
                int v0 = mesh_->to_vertex(e << 1);
                int v1 = mesh_->to_vertex(e << 1 | 1);
                vlpzsum[v0] = vlpzsum[v1] = T(0);
                if (calculate_lpzcnt) {
                    vlpzcnt[v0] = vlpzcnt[v1] = 0.0f;
                }
            }
        }
    }
    // only accumulate neighbours connected by a border edge for border vertices
    // same for feature vertices
    for (int e = 0; e < mesh_->lines_size_; ++e) {
        if (mesh_->has_garbage_ && edeleted[e])
            continue;
        if (mesh_->is_boundary_e(e) || efeature[e]) {
            int v0 = mesh_->to_vertex(e << 1);
            int v1 = mesh_->to_vertex(e << 1 | 1);
            vlpzsum[v0] += arr[v1];
            vlpzsum[v1] += arr[v0];
            if (calculate_lpzcnt) {
                vlpzcnt[v0] += 1.0f;
                vlpzcnt[v1] += 1.0f;
            }
        }
    }
}

static std::set<std::string> NO_LAPLACIAN({"v_normal", "v_sizing", "v_duplicate", "v_feature", "v_locked", "v_deleted"});

void SurfaceRemeshing::planar_laplacian(float delta) {
    auto &pos = mesh_->prim_->attr<vec3f>("pos");
    auto &vdeleted = mesh_->prim_->verts.attr<int>("v_deleted");
    auto &vlocked = mesh_->prim_->verts.attr<int>("v_locked");
    auto &vlpzcnt = mesh_->prim_->verts.add_attr<float>("v_lpzcnt", 0.0f);

    auto &vlpzsum = mesh_->prim_->verts.add_attr<vec3f>("v_lpzsum", vec3f(0.0f));
    accumulate_laplacian(pos, true, false);
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if ((mesh_->has_garbage_ && vdeleted[v]) || vlpzcnt[v] == 0)
            continue;
        vlpzsum[v] = (vlpzsum[v] + pos[v]) / (vlpzcnt[v] + 1);
    }
    for (int v = 0; v < mesh_->vertices_size_; ++v) {
        if ((mesh_->has_garbage_ && vdeleted[v]) || vlocked[v] || !vlpzcnt[v])
            continue;
        pos[v] = pos[v] * (1 - delta) + vlpzsum[v] * delta;
    }
    mesh_->prim_->verts.erase_attr("v_lpzsum");

    mesh_->prim_->verts.foreach_attr<zeno::AttrAcceptAll>([&](auto const &key, auto &arr) {
        if (NO_LAPLACIAN.count(key) == 0) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &vlpzsum = mesh_->prim_->verts.add_attr<T>("v_lpzsum", T(0));
            accumulate_laplacian(arr, false, true);
            for (int v = 0; v < mesh_->vertices_size_; ++v) {
                if ((mesh_->has_garbage_ && vdeleted[v]) || vlpzcnt[v] < 1e-5)
                    continue;
                vlpzsum[v] = (vlpzsum[v] + arr[v]) / (vlpzcnt[v] + 1);
            }

            for (int v = 0; v < mesh_->vertices_size_; ++v) {
                if ((mesh_->has_garbage_ && vdeleted[v]) || vlocked[v] || !vlpzcnt[v])
                    continue;
                arr[v] = arr[v] * (1 - delta) + vlpzsum[v] * delta;
            }
            mesh_->prim_->verts.erase_attr("v_lpzsum");
        }
    });

    mesh_->prim_->verts.erase_attr("v_lpzcnt");
}

} // namespace pmp
} // namespace zeno
