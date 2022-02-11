#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/orthonormal.h>
#include <zeno/utils/parallel.h>
#include <sstream>
#include <iostream>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace zeno {

struct PrimitiveBent : zeno::INode {
    virtual void apply() override {
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto angle = get_input<zeno::NumericObject>("angle")->get<float>();
        auto limitMin = get_input<zeno::NumericObject>("limitMin")->get<float>();
        auto limitMax = get_input<zeno::NumericObject>("limitMax")->get<float>();
        limitMin = std::min(1.f, std::max(0.f, limitMin));
        limitMax = std::min(1.f, std::max(0.f, limitMax));
        limitMin -= 0.5f;
        limitMax -= 0.5f;

        auto origin = has_input("origin") ? get_input<zeno::NumericObject>("origin")->get<vec3f>() : vec3f(0, 0, 0);
        auto tangent = has_input("tangent") ? get_input<zeno::NumericObject>("tangent")->get<vec3f>() : vec3f(0, 1, 0);
        auto direction = has_input("direction") ? get_input<zeno::NumericObject>("direction")->get<vec3f>() : vec3f(1, 0, 0);

        orthonormal orb(direction, tangent);
        direction = orb.normal;
        tangent = orb.tangent;

        if (std::abs(angle) > 0.005f && limitMax - limitMin > 0.001f) {
            angle *= M_PI / 180;
            angle /= limitMax - limitMin;

            auto acc = parallel_reduce_array(prim->size(), vec2f(prim->size() ? dot(tangent, prim->verts[0] - origin) : 0.f), [&] (size_t i) {
                return vec2f(dot(tangent, prim->verts[i] - origin));
            }, [&] (auto a, auto b) { return vec2f(std::min(a[0], b[0]), std::max(a[1], b[1])); });
            auto height = acc[1] - acc[0];
            auto middle = (acc[1] + acc[0]) * 0.5f;
            auto radius = height / angle;
            auto inv_height = 1 / height;

#pragma omp parallel for
            for (int i = 0; i < prim->verts.size(); i++) {
                auto pos = prim->verts[i] - origin;
                auto tanpos = dot(tangent, pos);
                auto dirpos = dot(direction, pos);
                auto fac = (tanpos - middle) * inv_height;

                auto rad = radius - dirpos;
                auto ang = std::max(limitMin, std::min(fac, limitMax)) * angle;
                auto sinang = std::sin(ang);
                auto cosang = std::cos(ang);
                auto newtanpos = rad * sinang;
                auto newdirpos = rad * cosang;

                auto diff = (std::max(0.f, limitMin - fac) + std::min(0.f, limitMax - fac)) * height;
                newtanpos -= diff * cosang;
                newdirpos += diff * sinang;

                newdirpos -= radius;
                pos += (newtanpos - tanpos) * tangent + (newdirpos - dirpos) * direction;

                prim->verts[i] = pos + origin;
            }

        }
        set_output("prim", std::move(prim));
    }
};


ZENDEFNODE(PrimitiveBent, {
    {
    {"PrimitiveObject", "prim"},
    {"vec3f", "origin", "0,0,0"},
    {"vec3f", "tangent", "0,1,0"},
    {"vec3f", "direction", "1,0,0"},
    {"float", "angle", "45"},
    {"float", "limitMin", "0"},
    {"float", "limitMax", "1"},
    },
    {
    {"PrimitiveObject", "prim"},
    },
    {
    },
    {"primitive"},
});

}
