#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/orthonormal.h>

namespace zeno {

struct AxisObject : IObjectClone<AxisObject> {
    vec3f origin, axisX, axisY, axisZ;

    AxisObject() : origin(0, 0, 0), axisX(1, 0, 0), axisY(0, 1, 0), axisZ(0, 0, 1) {}

    AxisObject(vec3f const &origin, vec3f const &axisX, vec3f const &axisY, vec3f const &axisZ)
        : origin(origin), axisX(axisX), axisY(axisY), axisZ(axisZ) {}

    void renormalizeByX() {
        orthonormal orb(axisX, axisY);
        axisX = orb.normal;
        axisY = orb.tangent;
        axisZ = orb.bitangent;
    }

    void renormalizeByY() {
        orthonormal orb(axisY, axisZ);
        axisY = orb.normal;
        axisZ = orb.tangent;
        axisX = orb.bitangent;
    }

    void renormalizeByZ() {
        orthonormal orb(axisZ, axisX);
        axisZ = orb.normal;
        axisX = orb.tangent;
        axisY = orb.bitangent;
    }

    vec3f transformPoint(vec3f const &v) {
        return origin + axisX * v[0] + axisY * v[1] + axisZ + v[2];
    }

    vec3f transformDirection(vec3f const &v) {
        return axisX * v[0] + axisY * v[1] + axisZ + v[2];
    }
};

} // namespace zeno
