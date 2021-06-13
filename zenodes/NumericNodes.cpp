#include <zen/zen.h>
#include <zen/NumericObject.h>
#include <iostream>


struct NumericInt : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::NumericObject>();
        obj->set(get_param<int>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(NumericInt, {
    {},
    {"value"},
    {{"int", "value", "0"}},
    {"numeric"},
});


struct NumericFloat : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::NumericObject>();
        obj->set(get_param<float>("value"));
        set_output("value", std::move(obj));
    }
};

ZENDEFNODE(NumericFloat, {
    {},
    {"value"},
    {{"float", "value", "0"}},
    {"numeric"},
});


struct NumericVec2 : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        obj->set(zen::vec2f(x, y));
        set_output("vec2", std::move(obj));
    }
};

ZENDEFNODE(NumericVec2, {
    {},
    {"vec2"},
    {{"float", "x", "0"}, {"float", "y", "0"}},
    {"numeric"},
});


struct NumericVec3 : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        obj->set(zen::vec3f(x, y, z));
        set_output("vec3", std::move(obj));
    }
};

ZENDEFNODE(NumericVec3, {
    {},
    {"vec3"},
    {{"float", "x", "0"}, {"float", "y", "0"}, {"float", "z", "0"}},
    {"numeric"},
});


struct NumericVec4 : zen::INode {
    virtual void apply() override {
        auto obj = std::make_unique<zen::NumericObject>();
        auto x = get_param<float>("x");
        auto y = get_param<float>("y");
        auto z = get_param<float>("z");
        auto w = get_param<float>("w");
        obj->set(zen::vec4f(x, y, z, w));
        set_output("vec4", std::move(obj));
    }
};

ZENDEFNODE(NumericVec4, {
    {},
    {"vec4"},
    {{"float", "x", "0"}, {"float", "y", "0"},
     {"float", "z", "0"}, {"float", "w", "0"}},
    {"numeric"},
});


struct PrintNumeric : zen::INode {
    template <class T>
    struct do_print {
        do_print(T const &x) {
            std::cout << x;
        }
    };

    template <size_t N, class T>
    struct do_print<zen::vec<N, T>> {
        do_print(zen::vec<N, T> const &x) {
            std::cout << "(";
            for (int i = 0; i < N; i++) {
                std::cout << x[i];
                if (i != N - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")";
        }
    };

    virtual void apply() override {
        auto obj = get_input<zen::NumericObject>("value");
        auto hint = get_param<std::string>("hint");
        std::cout << hint << ": ";
        std::visit([&](auto val) {
            do_print _(val);
        }, obj->value);
        std::cout << std::endl;
    }
};

ZENDEFNODE(PrintNumeric, {
    {"value"},
    {},
    {{"string", "hint", "PrintNumeric"}},
    {"numeric"},
});
