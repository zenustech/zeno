#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct CameraData {
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
