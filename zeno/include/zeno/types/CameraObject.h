#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct CameraData {
    vec3f pos{0, 0, 1};
    vec3f up{0, 1, 0};
    vec3f view{0, 0, -1};
    float fov{45.f};
    float fnear{0.1f};
    float ffar{20000.f};
    float nx{1920};
    float ny{1080};
    float dof{-1.f};
    float aperature{-1.f};
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
