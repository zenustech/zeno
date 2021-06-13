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


struct NumericOperator : zen::INode {

    struct op_add {
        template <class T1, class T2, decltype(
                std::declval<T1>() + std::declval<T2>()
                , true) = true>
        static auto apply(T1 const &x, T2 const &y) {
            return x + y;
        }
    };

    template <class T, class ...>
    using _left_t = T;

    template <class Op, class ...Ts>
    struct _op_apply {
        static int apply(Ts const &...ts) {
            std::cout << "Invalid numeric operation encountered!" << std::endl;
            return 0;
        }
    };

    template <class Op, class ...Ts>
    struct _op_apply<_left_t<Op, decltype(
            Op::apply(std::declval<Ts>()...))>, Ts...> {

        static auto apply(Ts const &...ts) {
            return Op::apply(ts...);
        }
    };

    template <class Op, class ...Ts>
    static auto op_apply(Ts const &...ts) {
        return _op_apply<Op, Ts...>::apply(ts...);
    }

    virtual void apply() override {
        auto op = get_param<std::string>("op_type");
        auto lhs = get_input<zen::NumericObject>("lhs");
        auto rhs = get_input<zen::NumericObject>("rhs");
        auto ret = std::make_unique<zen::NumericObject>();

        std::visit([op, &ret](auto const &lhs, auto const &rhs) {
            ret->value = op_apply<op_add>(lhs, rhs);
        }, lhs->value, rhs->value);

        set_output("ret", std::move(ret));
    }
};

ZENDEFNODE(NumericOperator, {
    {"lhs", "rhs"},
    {"ret"},
    {{"string", "op_type", "copy"}},
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
        std::visit([](auto const &val) {
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
