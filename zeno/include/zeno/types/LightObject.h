#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct LightData {
    //vec3f pos{1, 1, 0};
    vec3f lightDir{normalize(vec3f(1, 1, 0))};
    float intensity{10.0f};
    vec3f shadowTint{0.2f};
    float lightHight{1000.0f};
    float shadowSoftness{1.0f};
    vec3f lightColor{1.0f};
    float lightScale{1.0f};
    bool isEnabled{true};
};

struct LightObject : IObjectClone<LightObject>, LightData {
};

}
