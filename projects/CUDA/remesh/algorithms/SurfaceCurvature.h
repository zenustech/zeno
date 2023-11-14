// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <Eigen/Core>
#include <zeno/utils/vec.h>
#include "../SurfaceMesh.h"

namespace zeno {
namespace pmp {

class SurfaceCurvature {
public:
    SurfaceCurvature(SurfaceMesh* mesh);

    ~SurfaceCurvature();

    //! compute curvature information for each vertex, optionally followed
    //! by some smoothing iterations of the curvature values
    void analyze_tensor(unsigned int post_smoothing_steps = 0);

    //! return maximum absolute curvature
    float max_abs_curvature(int v) const {
        return std::max(fabs(min_curvature_[v]), fabs(max_curvature_[v]));
    }

    void calculate_gaussian_curvature() {
        auto &gaussian_curv = mesh_->prim_->verts.add_attr<float>("curv_gaussian");
        for (int v = 0; v < mesh_->vertices_size_; ++v) {
            if (mesh_->has_garbage_ && vdeleted[v])
                continue;
            gaussian_curv[v] = min_curvature_[v] * max_curvature_[v];
        }
    }

    bool symmetric_eigendecomposition(const Eigen::Matrix3f& m,
                                    float& eval1, float& eval2, float& eval3,
                                    vec3f& evec1, vec3f& evec2, vec3f& evec3) {
        unsigned int i, j;
        float theta, t, c, s;
        Eigen::Matrix3f V = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f R;
        Eigen::Matrix3f A = m;
        const float eps = 1e-10; //0.000001;

        int iterations = 100;
        while (iterations--) {
            // find largest off-diagonal elem
            if (fabs(A(0, 1)) < fabs(A(0, 2))) {
                if (fabs(A(0, 2)) < fabs(A(1, 2))) {
                    i = 1, j = 2;
                } else {
                    i = 0, j = 2;
                }
            } else {
                if (fabs(A(0, 1)) < fabs(A(1, 2))) {
                    i = 1, j = 2;
                } else {
                    i = 0, j = 1;
                }
            }

            // converged?
            if (fabs(A(i, j)) < eps)
                break;

            // compute Jacobi-Rotation
            theta = 0.5 * (A(j, j) - A(i, i)) / A(i, j);
            t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
            if (theta < 0.0)
                t = -t;

            c = 1.0 / sqrt(1.0 + t * t);
            s = t * c;

            R = Eigen::Matrix3f::Identity();
            R(i, i) = R(j, j) = c;
            R(i, j) = s;
            R(j, i) = -s;

            A = R.transpose() * A * R;
            V = V * R;
        }

        if (iterations > 0) {
            // sort and return
            int sorted[3];
            float d[3] = {A(0, 0), A(1, 1), A(2, 2)};

            if (d[0] > d[1]) {
                if (d[1] > d[2]) {
                    sorted[0] = 0, sorted[1] = 1, sorted[2] = 2;
                } else {
                    if (d[0] > d[2]) {
                        sorted[0] = 0, sorted[1] = 2, sorted[2] = 1;
                    } else {
                        sorted[0] = 2, sorted[1] = 0, sorted[2] = 1;
                    }
                }
            } else {
                if (d[0] > d[2]) {
                    sorted[0] = 1, sorted[1] = 0, sorted[2] = 2;
                } else {
                    if (d[1] > d[2]) {
                        sorted[0] = 1, sorted[1] = 2, sorted[2] = 0;
                    } else {
                        sorted[0] = 2, sorted[1] = 1, sorted[2] = 0;
                    }
                }
            }

            eval1 = d[sorted[0]];
            eval2 = d[sorted[1]];
            eval3 = d[sorted[2]];

            evec1 = vec3f(V(0, sorted[0]), V(1, sorted[0]), V(2, sorted[0]));
            evec2 = vec3f(V(0, sorted[1]), V(1, sorted[1]), V(2, sorted[1]));
            evec3 = normalize(cross(evec1, evec2));

            return true;
        }

        return false;
    }

private:
    //! smooth curvature values
    void smooth_curvatures(unsigned int iterations);

private:
    SurfaceMesh* mesh_;
    int vertice_num_;
    int edge_num_;
    AttrVector<float> min_curvature_;
    AttrVector<float> max_curvature_;
};

} // namespace pmp
} // namespace zeno
