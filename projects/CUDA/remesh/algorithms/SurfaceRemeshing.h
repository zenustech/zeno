// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <set>
#include <Eigen/Core>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/vec.h>
#include "../SurfaceMesh.h"

namespace zeno {
namespace pmp {

class TriangleKdTree;

// A class for uniform and adaptive surface remeshing.
class SurfaceRemeshing {
public:
    // Construct with mesh to be remeshed.
    SurfaceRemeshing(SurfaceMesh* mesh, std::string line_pick_tag, std::string length_tag);

    // destructor
    ~SurfaceRemeshing();

    // uniform remeshing with target edge length
    void uniform_remeshing(float edge_length,
                           unsigned int iterations = 10,
                           bool use_projection = true);

    // adaptive remeshing with min/max edge length and approximation error
    void adaptive_remeshing(float min_edge_length,
                            float max_edge_length,
                            float approx_error,
                            unsigned int iterations = 10,
                            bool use_projection = true);

    // remove degenerate triangles for mesh without remeshing
    // do not preserve feature vertices and edges
    void remove_degenerate_triangles(float min_edge_length,
                                     float min_area,
                                     float max_angle,
                                     std::string degenerate_tag,
                                     unsigned int iterations = 10,
                                     bool color = false);


private:
    void preprocessing();
    void postprocessing();

    int split_long_edges();
    enum COLLAPSE_COND {SHORT, CROSSES};
    void collapse_edges(COLLAPSE_COND cond);
    void degenerate_collapse_check(int v0, int v1,
                                   bool &h01, bool &h10,
                                   std::vector<int> &col_edges);
    void degenerate_collapse(const std::vector<int> &col_edges, int direction);
    void flip_edges();
    void tangential_smoothing(unsigned int iterations = 1);
    void laplacian_smoothing();
    void remove_caps(float max_angle = 170.f, bool try_collapse = false);
    void collapse_triangles(float min_edge_length, float min_area);
    void check_triangles(float min_edge_length = std::numeric_limits<float>::epsilon(),
                         float min_area = std::numeric_limits<float>::epsilon(),
                         float max_angle = 170.f,
                         std::string tag = "",
                         bool color = false);
    vec3f minimize_squared_areas(int v, bool& inversable);
    vec3f weighted_centroid(int v);
    template<class T>
    void accumulate_laplacian(std::vector<T>& arr,
                              bool calculate_lpzcnt = true,
                              bool only_feature = false);
    void planar_laplacian(float delta = 0.2);

    void project_to_reference(int v);

    bool is_too_long(int v0, int v1) const {
        auto& points = mesh_->prim_->attr<vec3f>("pos");
        auto& vsizing = mesh_->prim_->verts.attr<float>(length_tag_);
        return distance(points[v0], points[v1]) >
               4.0 / 3.0 * std::min(vsizing[v0], vsizing[v1]);
    }
    bool is_too_short(int v0, int v1) const {
        auto& points = mesh_->prim_->attr<vec3f>("pos");
        auto& vsizing = mesh_->prim_->verts.attr<float>(length_tag_);
        return distance(points[v0], points[v1]) <
               4.0 / 5.0 * std::min(vsizing[v0], vsizing[v1]);
    }
    bool is_crosses(int v0, int v1) const {
        int face_cnt = 0;
        for (auto ff : mesh_->faces(v0)) {
            ++face_cnt;
        }
        if (face_cnt == 3 || face_cnt == 4) return true;
        face_cnt = 0;
        for (auto ff : mesh_->faces(v1)) {
            ++face_cnt;
        }
        return (face_cnt == 3 || face_cnt == 4);
    }
    bool should_collapse(COLLAPSE_COND cond, int v0, int v1) const {
        if (cond == COLLAPSE_COND::SHORT) {
            return is_too_short(v0, v1);
        } else if (cond == COLLAPSE_COND::CROSSES) {
            return is_crosses(v0, v1);
        }
    }

private:
    SurfaceMesh* mesh_;
    SurfaceMesh* refmesh_;

    bool use_projection_;
    TriangleKdTree* kd_tree_;

    bool uniform_;
    float target_edge_length_;
    float min_edge_length_;
    float max_edge_length_;
    float approx_error_;
    std::string line_pick_tag_;
    std::string length_tag_;
    std::set<std::string> NO_LAPLACIAN_ = {"v_normal", "v_duplicate", "v_feature", "v_locked", "v_deleted"};


    AttrVector<vec3f> refpoints_;
    AttrVector<vec3f> refnormals_;
    AttrVector<float> refsizing_;

};

} // namespace pmp
} // namespace zeno
