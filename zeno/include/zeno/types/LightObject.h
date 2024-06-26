#pragma once

#ifndef __CUDACC_RTC__ 

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

#else

#ifndef vec3f
#define vec3f vec3
#endif

#endif

namespace zeno {

    enum struct LightType {
        Diffuse=0u, Direction=1u, IES=2u, Spot=3u, Projector=4u
    };

    enum struct LightShape {
        Plane=0u, Ellipse=1u, Sphere=2u, Point=3u, TriangleMesh=4u
    };

    enum LightConfigMask {
        LightConfigNull       = 0u,
        LightConfigVisible    = 1u,
        LightConfigDoubleside = 2u
    };

    struct DistantLightData {
        vec3f direction;
        float angle;
        vec3f color;
        float intensity;

        DistantLightData() = default;
    };

#ifndef __CUDACC_RTC__ 

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
    LightData const &get() const {
        return static_cast<LightData const &>(*this);
    }

    void set(LightData const &lit) {
        static_cast<LightData &>(*this) = lit;
    }
};

#endif
}
