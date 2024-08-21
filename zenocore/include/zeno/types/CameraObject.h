#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/gtx/quaternion.hpp>
#include <optional>

namespace zeno {
static glm::quat from_theta_phi(float theta, float phi) {
    float cos_t = glm::cos(theta), sin_t = glm::sin(theta);
    float cos_p = glm::cos(phi), sin_p = glm::sin(phi);
    glm::vec3 front(cos_t * sin_p, sin_t, -cos_t * cos_p);
    glm::vec3 up(-sin_t * sin_p, cos_t, sin_t * cos_p);
    glm::vec3 right = glm::cross(front, up);
    glm::mat3 rotation;
    rotation[0] = right;
    rotation[1] = up;
    rotation[2] = -front;
    return glm::quat_cast(rotation);
}

struct CameraData {
    vec3f pos{0, 0, 1};
    vec3f up{0, 1, 0};
    vec3f view{0, 0, -1};
    float fov{45.f};
    float fnear{0.01f};
    float ffar{20000.f};
    //float dof{-1.f};
    float aperture{0.0f};
    float focalPlaneDistance{2.0f};

    std::optional<vec3f> pivot = std::nullopt;
};

struct CameraObject : IObjectClone<CameraObject>, CameraData {
    void set(CameraData const &src) {
        static_cast<CameraData &>(*this) = src;
    }

    CameraData const &get() const {
        return static_cast<CameraData const &>(*this);
    }
};

}
