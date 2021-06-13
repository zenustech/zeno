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


    template <class T, class ...>
    using _left_t = T;

#define _PER_OP2(op, name) \
    template <class, class T1, class T2> \
    struct _op_##name { \
        static int apply(T1 const &t1, T2 const &t2) { \
            std::cout << "Invalid numeric operation encountered!" << std::endl; \
            return 0; \
        } \
    }; \
 \
    template <class T0, class T1, class T2> \
    struct _op_##name<_left_t<T0, decltype( \
            std::declval<T1>() op std::declval<T2>())>, T1, T2> { \
        static auto apply(T1 const &t1, T2 const &t2) { \
            return t1 op t2; \
        } \
    }; \
 \
    template <class T1, class T2> \
    static auto op_##name(T1 const &t1, T2 const &t2) { \
        return _op_##name<void, T1, T2>::apply(t1, t2); \
    }

#define _PER_OP1(op, name) \
    template <class, class T1> \
    struct _op_##name { \
        static int apply(T1 const &t1) { \
            std::cout << "Invalid numeric operation encountered!" << std::endl; \
            return 0; \
        } \
    }; \
 \
    template <class T0, class T1> \
    struct _op_##name<_left_t<T0, decltype( \
            op std::declval<T1>())>, T1> { \
        static auto apply(T1 const &t1) { \
            return op t1; \
        } \
    }; \
 \
    template <class T1> \
    static auto op_##name(T1 const &t1) { \
        return _op_##name<void, T1>::apply(t1); \
    }

#define _PER_FN(name) \
    template <class, class ...Ts> \
    struct _op_##name { \
        static int apply(Ts const &...ts) { \
            std::cout << "Invalid numeric operation encountered!" << std::endl; \
            return 0; \
        } \
    }; \
 \
    template <class T0, class ...Ts> \
    struct _op_##name<_left_t<T0, decltype( \
            zen::name(std::declval<Ts>()...))>, Ts...> { \
        static auto apply(Ts const &...ts) { \
            return zen::name(ts...); \
        } \
    }; \
 \
    template <class ...Ts> \
    static auto op_##name(Ts const &...ts) { \
        return _op_##name<void, Ts...>::apply(ts...); \
    }

    _PER_OP2(+, add)
    _PER_OP2(-, sub)
    _PER_OP2(*, mul)
    _PER_OP2(/, div)
    _PER_OP2(%, mod)
    _PER_OP2(&, and)
    _PER_OP2(|, or)
    _PER_OP2(^, xor)
    _PER_OP2(>>, shr)
    _PER_OP2(<<, shl)

    _PER_OP1(+, pos)
    _PER_OP1(-, neg)
    _PER_OP1(~, inv)
    _PER_OP1(!, not)

    _PER_FN(atan2)
    _PER_FN(pow)
    _PER_FN(max)
    _PER_FN(min)
    _PER_FN(fmod)

    _PER_FN(abs)
    _PER_FN(sqrt)
    _PER_FN(sin)
    _PER_FN(cos)
    _PER_FN(tan)
    _PER_FN(asin)
    _PER_FN(acos)
    _PER_FN(atan)
    _PER_FN(exp)
    _PER_FN(log)
    _PER_FN(floor)
    _PER_FN(ceil)

#undef _PER_FN
#undef _PER_OP2
#undef _PER_OP1

    virtual void apply() override {
        auto op = get_param<std::string>("op_type");
        auto ret = std::make_unique<zen::NumericObject>();
        auto lhs = get_input<zen::NumericObject>("lhs");

        if (has_input("rhs")) {
            auto rhs = get_input<zen::NumericObject>("rhs");

            std::visit([op, &ret](auto const &lhs, auto const &rhs) {

                if (op == "copy") ret->value = lhs;
                else if (op == "copyr") ret->value = rhs;
#define _PER_OP(name) else if (op == #name) ret->value = op_##name(lhs, rhs);
    _PER_OP(add)
    _PER_OP(sub)
    _PER_OP(mul)
    _PER_OP(div)
    _PER_OP(mod)
    _PER_OP(and)
    _PER_OP(or)
    _PER_OP(xor)
    _PER_OP(shr)
    _PER_OP(shl)
    _PER_OP(atan2)
    _PER_OP(pow)
    _PER_OP(max)
    _PER_OP(min)
    _PER_OP(fmod)
#undef _PER_OP

            }, lhs->value, rhs->value);

        } else {
            std::visit([op, &ret](auto const &lhs) {

                if (op == "copy" || op == "copyr") ret->value = lhs;
#define _PER_OP(name) else if (op == #name) ret->value = op_##name(lhs);
    _PER_OP(pos)
    _PER_OP(neg)
    _PER_OP(inv)
    _PER_OP(not)
    _PER_OP(abs)
    _PER_OP(sqrt)
    _PER_OP(sin)
    _PER_OP(cos)
    _PER_OP(tan)
    _PER_OP(asin)
    _PER_OP(acos)
    _PER_OP(atan)
    _PER_OP(exp)
    _PER_OP(log)
    _PER_OP(floor)
    _PER_OP(ceil)
#undef _PER_OP

            }, lhs->value);
        }

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
