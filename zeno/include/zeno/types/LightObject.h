#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct LightData {
    vec3f pos{1, 1, 0};
    vec3f dir{-1, -1, 0};
    float intensity{100};
};

struct LightObject : IObjectClone<LightObject>, LightData {
};

}
