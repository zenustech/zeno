#pragma once
#include <cstdio>
#include <Eigen/Core>
#include <vector>
#include "attribute.h"
#include "context.h"
#include "kernels/query_dispatcher.cuh"

namespace zeno::rxmesh {
__device__ __inline__ float update_step(
    const rxmesh::ElementHandle&       v0_id,
    const rxmesh::ElementHandle&       v1_id,
    const rxmesh::ElementHandle&       v2_id,
    const rxmesh::VertexAttribute<float>& geo_distance,
    const rxmesh::VertexAttribute<float>& coords,
    const float                           infinity_val) {

    using namespace rxmesh;

    Eigen::Vector3f v0(coords(v0_id, 0), coords(v0_id, 1), coords(v0_id, 2));
    Eigen::Vector3f v1(coords(v1_id, 0), coords(v1_id, 1), coords(v1_id, 2));
    Eigen::Vector3f v2(coords(v2_id, 0), coords(v2_id, 1), coords(v2_id, 2));

    const Eigen::Vector3f x0 = v1 - v0;
    const Eigen::Vector3f x1 = v2 - v0;

    float t[2];
    t[0] = geo_distance(v1_id);
    t[1] = geo_distance(v2_id);

    float q[2][2];

    q[0][0] = x0.dot(x0);
    q[0][1] = x0.dot(x1);
    q[1][0] = x1.dot(x0);
    q[1][1] = x1.dot(x1);


    float det = q[0][0] * q[1][1] - q[0][1] * q[1][0];
    float Q[2][2];
    Q[0][0] = q[1][1] / det;
    Q[0][1] = -q[0][1] / det;
    Q[1][0] = -q[1][0] / det;
    Q[1][1] = q[0][0] / det;

    float delta = t[0] * (Q[0][0] + Q[1][0]) + t[1] * (Q[0][1] + Q[1][1]);
    float dis   = delta * delta -
            (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]) *
                (t[0] * t[0] * Q[0][0] + t[0] * t[1] * (Q[1][0] + Q[0][1]) +
                 t[1] * t[1] * Q[1][1] - 1);
    float p = (delta + std::sqrt(dis)) / (Q[0][0] + Q[0][1] + Q[1][0] + Q[1][1]);
    float tp[2];
    tp[0]                = t[0] - p;
    tp[1]                = t[1] - p;
    const Eigen::Vector3f n = (x0 * Q[0][0] + x1 * Q[1][0]) * tp[0] +
                           (x0 * Q[0][1] + x1 * Q[1][1]) * tp[1];
    float cond[2];
    cond[0] = x0.dot(n);
    cond[1] = x1.dot(n);

    float c[2];
    c[0] = cond[0] * Q[0][0] + cond[1] * Q[0][1];
    c[1] = cond[0] * Q[1][0] + cond[1] * Q[1][1];

    if (t[0] == infinity_val || t[1] == infinity_val || dis < 0 || c[0] >= 0 ||
        c[1] >= 0) {
        float dp[2];
        dp[0] = geo_distance(v1_id) + x0.norm();
        dp[1] = geo_distance(v2_id) + x1.norm();
        p     = dp[dp[1] < dp[0]];
    }
    return p;
}


template <uint32_t blockThreads>
__global__ static void relax_ptp_rxmesh(
    const rxmesh::Context                   context,
    const rxmesh::VertexAttribute<float>    coords,
    rxmesh::VertexAttribute<float>          new_geo_dist,
    const rxmesh::VertexAttribute<float>    old_geo_dist,
    const rxmesh::VertexAttribute<uint32_t> toplesets,
    const uint32_t                          band_start,
    const uint32_t                          band_end,
    uint32_t*                               d_error,
    const float                             infinity_val,
    const float                             error_tol) {
    using namespace rxmesh;
    auto in_active_set = [&](ElementHandle p_id) {
        return true;
        // uint32_t my_band = toplesets(p_id);
        // return my_band >= band_start && my_band < band_end;
    };

    auto geo_lambda = [&](ElementHandle& p_id, const ElementIterator& iter) {
        // this vertex (p_id) update_band
        uint32_t my_band = toplesets(p_id);

        // this is the last vertex in the one-ring (before r_id)
        auto q_id = iter.back();

        // one-ring enumeration
        float current_dist = old_geo_dist(p_id);
        float new_dist     = current_dist;
        for (uint32_t v = 0; v < iter.size(); ++v) {
            // the current one ring vertex
            auto r_id = iter[v];

            float dist = update_step(
                p_id, q_id, r_id, old_geo_dist, coords, infinity_val);
            if (dist < new_dist) {
                new_dist = dist;
            }
            q_id = r_id;
        }

        new_geo_dist(p_id) = new_dist;
        // update our distance
        if (my_band == band_start) {
            float error = fabs(new_dist - current_dist) / current_dist;
            if (error < error_tol) {
                ::atomicAdd(d_error, 1);
            }
        }
    };
    query_block_dispatcher<Op::VV, blockThreads>(
        context, geo_lambda, in_active_set, true);
}
}