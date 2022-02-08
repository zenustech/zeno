#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
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

        auto origin = has_input("origin") ? get_input<zeno::NumericObject>("origin")->get<vec3f>() : vec3f(0, 0, 0);
        auto tangent = has_input("tangent") ? get_input<zeno::NumericObject>("tangent")->get<vec3f>() : vec3f(0, 1, 0);
        auto direction = has_input("direction") ? get_input<zeno::NumericObject>("direction")->get<vec3f>() : vec3f(1, 0, 0);
        tangent = normalize(tangent);
        direction = normalize(direction);

        if (std::abs(angle) > 0.005f) {
            angle *= M_PI / 90;

            auto acc = parallel_reduce_array(prim->size(), vec2f(prim->size() ? dot(tangent, prim->verts[0]) : 0.f), [&] (size_t i) {
                return vec2f(dot(tangent, prim->verts[i]));
            }, [&] (auto a, auto b) { return vec2f(std::min(a[0], b[0]), std::max(a[1], b[1])); });
            auto height = acc[1] - acc[0];
            auto radius = height * 2 / angle;
            auto inv_radius = 1 / radius;
            //printf("%f %f\n", height, radius);

#pragma omp parallel for
            for (int i = 0; i < prim->verts.size(); i++) {
                auto pos = prim->verts[i];
                auto tanpos = dot(tangent, pos);
                auto dirpos = dot(direction, pos);

                auto rad = radius - dirpos;
                auto ang = tanpos * inv_radius;
                auto newtanpos = rad * std::sin(ang);
                auto newdirpos = rad * std::cos(ang) - radius;

                pos += (newtanpos - tanpos) * tangent + (newdirpos - dirpos) * direction;
                prim->verts[i] = pos;
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
