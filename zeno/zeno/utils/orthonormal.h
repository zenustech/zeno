#pragma once

#include <zeno/utils/vec.h>
#include <tuple>

namespace zeno {

struct orthonormal {
    vec3f normal, tangent, bitangent;

    explicit orthonormal(vec3f const &normal_)
        : normal(normal_) {
        normal = normalize(normal);
        tangent = vec3f(0, 0, 1);
        bitangent = cross(normal, tangent);
        if (dot(bitangent, bitangent) < 1e-5) {
            tangent = vec3f(0, 1, 0);
            bitangent = cross(normal, tangent);
        }
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);
        tangent = normalize(tangent);
    }

    explicit orthonormal(vec3f const &normal_, vec3f const &tangent_)
        : normal(normal_), tangent(tangent_) {
        bitangent = cross(normal, tangent);
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);
        tangent = normalize(tangent);
    }
};

}
