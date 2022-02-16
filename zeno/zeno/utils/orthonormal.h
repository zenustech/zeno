#pragma once

#include <zeno/utils/vec.h>
#include <tuple>

namespace zeno {

struct orthonormal {
    vec3f normal, tangent, bitangent;

    explicit orthonormal(vec3f const &normal_)
        : normal(normal_) {
        normal = normalize(normal);

        // https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
        if (normal[2] < -0.9999999f) {
            tangent = vec3f(0.0f, -1.0f, 0.0f);
            bitangent = vec3f(-1.0f, 0.0f, 0.0f);
            return;
        }
        float a = 1.0f / (1.0f + normal[2]);
        float b = -normal[0]*normal[1]*a;
        tangent = vec3f(1.0f - normal[0]*normal[0]*a, b, -normal[0]);
        bitangent = vec3f(b, 1.0f - normal[1]*normal[1]*a, -normal[1]);
    }

    explicit orthonormal(vec3f const &normal_, vec3f const &tangent_)
        : normal(normal_), tangent(tangent_) {
        bitangent = cross(normal, tangent);
        bitangent = normalize(bitangent);
        tangent = cross(bitangent, normal);
        //tangent = normalize(tangent);
    }
};

}
