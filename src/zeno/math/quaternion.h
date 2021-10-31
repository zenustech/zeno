#pragma once


#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace math {


struct quaternion_matrix {
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;

    constexpr quaternion_matrix(math::vec4f const &q) {
        // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        auto s = 2 / dot(q, q);
        auto [qi, qj, qk, qr] = std::make_tuple(q[0], q[1], q[2], q[3]);
        m11 = 1 - s * (qj*qj + qk*qk);
        m12 = s * (qi*qj - qk*qr);
        m13 = s * (qi*qk + qj*qr);
        m21 = s * (qi*qj + qk*qr);
        m22 = 1 - s * (qi*qi + qk*qk);
        m23 = s * (qj*qk - qi*qr);
        m31 = s * (qi*qk - qj*qr);
        m32 = s * (qj*qk + qi*qr);
        m33 = 1 - s * (qi*qi + qj*qj);
    }

    constexpr math::vec3f operator*(math::vec3f const &v) const {
        auto [vi, vj, vk] = std::make_tuple(v[0], v[1], v[2]);
        return {
            m11 * vi + m12 * vj + m13 * vk,
            m21 * vi + m22 * vj + m23 * vk,
            m31 * vi + m32 * vj + m33 * vk,
        };
    }
};


}
ZENO_NAMESPACE_END
