#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

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

    bool isSet = false;
    vec3f center{0, 0, 0};
    float radius{1};
    float theta{};
    float phi{};
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
