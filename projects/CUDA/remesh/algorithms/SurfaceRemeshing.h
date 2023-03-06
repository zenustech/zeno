// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <Eigen/Core>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/vec.h>
#include "../SurfaceMesh.h"

namespace zeno {
namespace pmp {

class TriangleKdTree;

//! A class for uniform and adaptive surface remeshing.
class SurfaceRemeshing {
public:
    //! Construct with mesh to be remeshed.
    SurfaceRemeshing(SurfaceMesh* mesh);

    // destructor
    ~SurfaceRemeshing();

    //! uniform remeshing with target edge length
    void uniform_remeshing(float edge_length, unsigned int iterations = 10,
                           bool use_projection = true);

    //! adaptive remeshing with min/max edge length and approximation error
    void adaptive_remeshing(float min_edge_length, float max_edge_length,
                            float approx_error, unsigned int iterations = 10,
                            bool use_projection = true);


private:
    void preprocessing();
    void postprocessing();

    void split_long_edges();
    void collapse_short_edges();
    void flip_edges();
    void tangential_smoothing(unsigned int iterations);
    void remove_caps();

    vec3f minimize_squared_areas(int v, bool& inversable);
    vec3f weighted_centroid(int v);

    void project_to_reference(int v);

    bool is_too_long(int v0, int v1) const {
        auto& points = mesh_->prim_->attr<vec3f>("pos");
        auto& vsizing = mesh_->prim_->verts.attr<float>("v_sizing");
        return distance(points[v0], points[v1]) >
               4.0 / 3.0 * std::min(vsizing[v0], vsizing[v1]);
    }
    bool is_too_short(int v0, int v1) const {
        auto& points = mesh_->prim_->attr<vec3f>("pos");
        auto& vsizing = mesh_->prim_->verts.attr<float>("v_sizing");
        return distance(points[v0], points[v1]) <
               4.0 / 5.0 * std::min(vsizing[v0], vsizing[v1]);
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


    AttrVector<vec3f> refpoints_;
    AttrVector<vec3f> refnormals_;
    AttrVector<float> refsizing_;

};

} // namespace pmp
} // namespace zeno
