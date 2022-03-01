#include <zeno/zeno.h>
#include <zeno/types/NumericObject.h>
#include <iostream>

namespace {

template <size_t N>
auto remove_bool(zeno::vec<N, bool> const &v) {
    return zeno::vec<N, int>(v);
}

template <class T>
decltype(auto) remove_bool(T const &t) {
    return t;
}

struct NumericOperator : zeno::INode {

    template <class T, class ...>
    using _left_t = T;

#define _PER_OP2(op, name) \
    template <class, class T1, class T2> \
    struct _op_##name { \
        static int apply(T1 const &t1, T2 const &t2) { \
            throw zeno::Exception("Invalid numeric operation encountered!"); \
        } \
    }; \
 \
    template <class T1, class T2> \
    struct _op_##name<std::void_t<decltype( \
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
            throw zeno::Exception("Invalid numeric operation encountered!"); \
        } \
    }; \
 \
    template <class T1> \
    struct _op_##name<std::void_t<decltype( \
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
            throw zeno::Exception("Invalid numeric operation encountered!"); \
        } \
    }; \
 \
    template <class ...Ts> \
    struct _op_##name<std::void_t<decltype( \
            zeno::name(std::declval<Ts>()...))>, Ts...> { \
        static auto apply(Ts const &...ts) { \
            return zeno::name(ts...); \
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
    _PER_OP2(>=, cmpge)
    _PER_OP2(<=, cmple)
    _PER_OP2(>, cmpgt)
    _PER_OP2(<, cmplt)
    _PER_OP2(!=, cmpne)
    _PER_OP2(==, cmpeq)
    _PER_OP2(&&, land)
    _PER_OP2(||, lor)

    _PER_OP1(+, pos)
    _PER_OP1(-, neg)
    _PER_OP1(~, inv)
    _PER_OP1(!, not)

    //_PER_FN(mix)
    //_PER_FN(clamp)

    _PER_FN(atan2)
    _PER_FN(pow)
    _PER_FN(max)
    _PER_FN(min)
    _PER_FN(fmod)
    _PER_FN(dot)
    _PER_FN(cross)
    _PER_FN(distance)

    _PER_FN(length)
    _PER_FN(normalize)
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
    _PER_FN(toint)
    _PER_FN(tofloat)
    _PER_FN(anytrue)
    _PER_FN(alltrue)

#undef _PER_FN
#undef _PER_OP2
#undef _PER_OP1

    virtual void apply() override {
        auto op = get_param<std::string>("op_type");
        auto ret = std::make_unique<zeno::NumericObject>();
        auto lhs = get_input<zeno::NumericObject>("lhs");
        auto rhs = has_input("rhs") ?
            get_input<zeno::NumericObject>("rhs")
            : std::make_shared<zeno::NumericObject>(0);
        
        // todo: no ternary ops..
        std::visit([op, &ret](auto const &lhs, auto const &rhs) {

            if (op == "copy") ret->value = remove_bool(lhs);
            if (op == "copyr") lhs->value = remove_bool(rhs->value);
#define _PER_OP(name) else if (op == #name) ret->value = remove_bool(op_##name(lhs));
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
_PER_OP(length)
_PER_OP(normalize)
_PER_OP(toint)
_PER_OP(tofloat)
_PER_OP(anytrue)
_PER_OP(alltrue)
#undef _PER_OP
#define _PER_OP(name) else if (op == #name) ret->value = remove_bool(op_##name(lhs, rhs));
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
_PER_OP(cmpge)
_PER_OP(cmple)
_PER_OP(cmpgt)
_PER_OP(cmplt)
_PER_OP(cmpne)
_PER_OP(cmpeq)
_PER_OP(land)
_PER_OP(lor)
_PER_OP(dot)
_PER_OP(cross)
_PER_OP(distance)
            else throw zeno::Exception("Bad op name: " + op);
#undef _PER_OP

        }, lhs->value, rhs->value);

        set_output("ret", std::move(ret));
    }
};

ZENO_DEFNODE(NumericOperator)({
    {{"NumericObject", "lhs"}, {"NumericObject", "rhs"}},
    {{"NumericObject", "ret"}},
    {{"enum"
#define _PER_FN(x) " " #x
    _PER_FN(add)
    _PER_FN(sub)
    _PER_FN(mul)
    _PER_FN(div)
    _PER_FN(mod)
    _PER_FN(and)
    _PER_FN(or)
    _PER_FN(xor)
    _PER_FN(shr)
    _PER_FN(shl)
    _PER_FN(cmpge)
    _PER_FN(cmple)
    _PER_FN(cmpgt)
    _PER_FN(cmplt)
    _PER_FN(cmpne)
    _PER_FN(cmpeq)
    _PER_FN(land)
    _PER_FN(lor)

    _PER_FN(pos)
    _PER_FN(neg)
    _PER_FN(inv)
    _PER_FN(not)

    //_PER_FN(mix)
    //_PER_FN(clamp)

    _PER_FN(atan2)
    _PER_FN(pow)
    _PER_FN(max)
    _PER_FN(min)
    _PER_FN(fmod)
    _PER_FN(dot)
    _PER_FN(cross)
    _PER_FN(distance)

    _PER_FN(length)
    _PER_FN(normalize)
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
    _PER_FN(toint)
    _PER_FN(tofloat)
    _PER_FN(anytrue)
    _PER_FN(alltrue)

    _PER_FN(copy)
    _PER_FN(copyr)
#undef _PER_FN
    , "op_type", "add"}},
    {"numeric"},
});

}
