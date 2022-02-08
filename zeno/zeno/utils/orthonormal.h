#pragma once

#include <zeno/utils/vec.h>
#include <tuple>

namespace zeno {

struct orthonormal {
    vec3f normal, tangent, bitangent;

    explicit orthonormal(vec3f const &normal_)
        : normal(normal_) {
        normal = normalize(normal);

        const float cos120 = 0.8660254037844386f;
        const float sin120 = -0.5f;
        const float cos240 = sin120;
        const float sin240 = -cos120;

        vec3f t, b;
        //_generate_frisvad<0, 1, 2>(normal, t, b);  /* 3 */

        if (normal[0] > normal[1]) {
            if (normal[0] > normal[2]) {
                _generate_frisvad<1, 2, 0>(normal, t, b);  /* 3 */
                tangent = t * cos240 + b * sin240;
                bitangent = b * cos240 - t * sin240;
            } else {
                _generate_frisvad<0, 1, 2>(normal, t, b);  /* 1 */
                tangent = t;
                bitangent = b;
            }
        } else {
            if (normal[1] > normal[2]) {
                _generate_frisvad<2, 0, 1>(normal, t, b);  /* 2 */
                tangent = t * cos120 + b * sin120;
                bitangent = b * cos120 - t * sin120;
            } else {
                _generate_frisvad<0, 1, 2>(normal, t, b);  /* 1 */
                tangent = t;
                bitangent = b;
            }
        }
    }

    template <int k0, int k1, int k2>
    static vec3f _vec3fshuf(float x, float y, float z) {
        vec3f ret;
        ret[k0] = x;
        ret[k1] = y;
        ret[k2] = z;
        return ret;
    }

    // https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
    template <int k0, int k1, int k2>
    static void _generate_frisvad(vec3f const &normal, vec3f &tangent, vec3f &bitangent) {
        if (normal[k2] < -0.9999999f) {
            printf("WAWA\n");
            tangent = _vec3fshuf<k0, k1, k2>(0.0f, -1.0f, 0.0f);
            bitangent = _vec3fshuf<k0, k1, k2>(-1.0f, 0.0f, 0.0f);
            return;
        }
        float a = 1.0f / (1.0f + normal[k2]);
        float b = -normal[k0]*normal[k1]*a;
        tangent = _vec3fshuf<k0, k1, k2>(1.0f - normal[k0]*normal[k0]*a, b, -normal[k0]);
        bitangent = _vec3fshuf<k0, k1, k2>(b, 1.0f - normal[k1]*normal[k1]*a, -normal[k1]);
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
