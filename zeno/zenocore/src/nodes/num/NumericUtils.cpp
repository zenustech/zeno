#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <zeno/utils/random.h>

namespace zeno {
namespace {

struct UnpackNumericVec : INode {
    virtual void apply() override {
        auto vec = get_input<NumericObject>("vec")->value;
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
        set_output("X", std::make_shared<NumericObject>(x));
        set_output("Y", std::make_shared<NumericObject>(y));
        set_output("Z", std::make_shared<NumericObject>(z));
        set_output("W", std::make_shared<NumericObject>(w));
    }
};

ZENDEFNODE(UnpackNumericVec, {
    {{"vec3f", "vec"}},
    {{"float", "X"}, {"float", "Y"},
     {"float", "Z"}, {"float", "W"}},
    {},
    {"numeric"},
}); // TODO: add PackNumericVec too.


struct NumericRandom : INode {
    virtual void apply() override {
        auto value = std::make_shared<NumericObject>();
        auto dim = get_param<int>("dim");
        auto symmetric = get_param<bool>("symmetric");
        auto scale = has_input("scale") ?
            get_input<NumericObject>("scale")->get<float>()
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
        set_output("value", std::move(value));
    }
};

ZENDEFNODE(NumericRandom, {
    {{"float", "scale", "1"}},
    {{"NumericObject", "value"}},
    {{"int", "dim", "1"}, {"bool", "symmetric", "0"}},
    {"deprecated"},
});


struct NumericRandomInt : INode {
    virtual void apply() override {
        auto value = std::make_shared<NumericObject>();
        auto minVal = has_input("min") ?
            get_input<NumericObject>("min")->get<int>()
            : 0;
        auto maxVal = has_input("max") ?
            get_input<NumericObject>("max")->get<int>()
            : 65536;
        value->set(int(rand()) % (maxVal - minVal) + minVal);
        set_output("value", std::move(value));
    }
};

ZENDEFNODE(NumericRandomInt, {
    {{"int", "min", "0"}, {"int", "max", "65536"}},
    {{"int", "value"}},
    {},
    {"deprecated"},
});


struct SetRandomSeed : INode {
    virtual void apply() override {
        auto seed = get_input<NumericObject>("seed")->get<int>();
        sfrand(seed);
        if (has_input("routeIn")) {
            set_output("routeOut", get_input("routeIn"));
        } else {
            set_output("routeOut", std::make_shared<NumericObject>(seed));
        }
    }
};

ZENDEFNODE(SetRandomSeed, {
    {"routeIn", {"int", "seed", "0"}},
    {"routeOut"},
    {},
    {"deprecated"},
});


struct NumericCounter : INode {
    int counter = 0;

    virtual void apply() override {
        auto count = std::make_shared<NumericObject>();
        count->value = counter++;
        set_output("count", std::move(count));
    }
};

ZENDEFNODE(NumericCounter, {
    {},
    {{"int", "count"}},
    {},
    {"numeric"},
});

}
}
