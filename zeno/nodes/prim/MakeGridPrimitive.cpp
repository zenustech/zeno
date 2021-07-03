#include <zeno/zeno.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/NumericObject.h>
#include <zeno/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

namespace zeno {


struct Make2DGridPrimitive : INode {
    virtual void apply() override {
        size_t nx = get_input<NumericObject>("nx")->get<int>();
        size_t ny = has_input("ny") ?
            get_input<NumericObject>("ny")->get<int>() : nx;
        float dx = 1.f / std::max(nx - 1, (size_t)1);
        float dy = 1.f / std::max(ny - 1, (size_t)1);
        vec3f ax = get_input<NumericObject>("sizeX")->get<vec3f>() * dx;
        vec3f ay = get_input<NumericObject>("sizeY")->get<vec3f>() * dy;
        vec3f o = has_input("origin") ?
            get_input<NumericObject>("origin")->get<vec3f>() : vec3f(0);
        if (has_input("scale")) {
            auto scale = get_input<NumericObject>("scale")->get<float>();
            ax *= scale;
            ay *= scale;
        }

        if (get_param<int>("isCentered"))
            o -= (ax * (nx - 1) + ay * (ny - 1)) / 2;

        auto prim = std::make_shared<PrimitiveObject>();
        prim->resize(nx * ny);
        auto &pos = prim->add_attr<vec3f>("pos");
        for (size_t y = 0; y < ny; y++) {
            for (size_t x = 0; x < nx; x++) {
                vec3f p = o + x * ax + y * ay;
                size_t i = x + y * nx;
                pos[i] = p;
            }
        }

        set_output("prim", std::move(prim));
    }
};

ZENDEFNODE(Make2DGridPrimitive,
        { /* inputs: */ {
        "nx", "ny", "sizeX", "sizeY", "scale", "origin",
        }, /* outputs: */ {
        "prim",
        }, /* params: */ {
        {"int", "isCentered", "0"},
        }, /* category: */ {
        "primitive",
        }});


}
