#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/random.h>

namespace zeno {
namespace {

struct UnpackNumericVec : INode {
    virtual void apply() override {
        auto vec = ZImpl(get_input<NumericObject>("vec"))->value;
        NumericValue x = 0, y = 0, z = 0, w = 0;
        std::visit([&x, &y, &z, &w] (auto const &vec) {
            using T = std::decay_t<decltype(vec)>;
            if constexpr (!is_vec_v<T>) {
                x = vec;
            } else {
                if constexpr (is_vec_n<T> > 0) x = vec[0];
                if constexpr (is_vec_n<T> > 1) y = vec[1];
                if constexpr (is_vec_n<T> > 2) z = vec[2];
                if constexpr (is_vec_n<T> > 3) w = vec[3];
            }
        }, vec);
        ZImpl(set_output("X", std::make_unique<NumericObject>(x)));
        ZImpl(set_output("Y", std::make_unique<NumericObject>(y)));
        ZImpl(set_output("Z", std::make_unique<NumericObject>(z)));
        ZImpl(set_output("W", std::make_unique<NumericObject>(w)));
    }
};

ZENDEFNODE(UnpackNumericVec, {
    {{gParamType_Vec3f, "vec"}},
    {{gParamType_Float, "X"}, {gParamType_Float, "Y"},
     {gParamType_Float, "Z"}, {gParamType_Float, "W"}},
    {},
    {"numeric"},
}); // TODO: add PackNumericVec too.


struct NumericRandom : INode {
    virtual void apply() override {
        auto value = std::make_unique<NumericObject>();
        auto dim = ZImpl(get_param<int>("dim"));
        auto symmetric = ZImpl(get_param<bool>("symmetric"));
        auto scale = ZImpl(has_input("scale")) ?
            ZImpl(get_input<NumericObject>("scale"))->get<float>()
            : 1.0f;
        float offs = 0.0f;
        if (symmetric) {
            offs = -scale;
            scale *= 2.0f;
        }
        if (dim == 1) {
            value->set(offs + scale * float(frand()));
        } else if (dim == 2) {
            value->set(offs + scale * zeno::vec2f(frand(), frand()));
        } else if (dim == 3) {
            value->set(offs + scale * zeno::vec3f(frand(), frand(), frand()));
        } else if (dim == 4) {
            value->set(offs + scale * zeno::vec4f(frand(), frand(), frand(), frand()));
        } else {
            char buf[1024];
            sprintf(buf, "invalid dim for NumericRandom: %d\n", dim);
            throw Exception(buf);
        }
        ZImpl(set_output("value", std::move(value)));
    }
};

ZENDEFNODE(NumericRandom, {
    {{gParamType_Float, "scale", "1"}},
    {{gParamType_Float, "value", "1"}},
    {{gParamType_Int, "dim", "1"}, {gParamType_Bool, "symmetric", "0"}},
    {"deprecated"},
});


struct NumericRandomInt : INode {
    virtual void apply() override {
        auto value = std::make_unique<NumericObject>();
        auto minVal = ZImpl(has_input("min")) ?
            ZImpl(get_input<NumericObject>("min"))->get<int>()
            : 0;
        auto maxVal = ZImpl(has_input("max")) ?
            ZImpl(get_input<NumericObject>("max"))->get<int>()
            : 65536;
        value->set(int(rand()) % (maxVal - minVal) + minVal);
        ZImpl(set_output("value", std::move(value)));
    }
};

ZENDEFNODE(NumericRandomInt, {
    {{gParamType_Int, "min", "0"}, {gParamType_Int, "max", "65536"}},
    {{gParamType_Int, "value"}},
    {},
    {"deprecated"},
});


struct SetRandomSeed : INode {
    virtual void apply() override {
        auto seed = ZImpl(get_input<NumericObject>("seed"))->get<int>();
        sfrand(seed);
        if (ZImpl(has_input("routeIn"))) {
            ZImpl(set_output("routeOut", ZImpl(clone_input("routeIn"))));
        } else {
            ZImpl(set_output("routeOut", std::make_unique<NumericObject>(seed)));
        }
    }
};

ZENDEFNODE(SetRandomSeed, {
    {{"object", "routeIn"}, {gParamType_Int, "seed", "0"}},
    {{"NumericObject","routeOut"}},
    {},
    {"deprecated"},
});


struct NumericCounter : INode {
    int counter = 0;

    virtual void apply() override {
        auto count = std::make_unique<NumericObject>();
        count->value = counter++;
        ZImpl(set_output("count", std::move(count)));
    }
};

ZENDEFNODE(NumericCounter, {
    {},
    {{gParamType_Int, "count"}},
    {},
    {"numeric"},
});

}
}
