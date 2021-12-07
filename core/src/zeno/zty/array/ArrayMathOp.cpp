#include <zeno/zty/array/Array.h>
#include <zeno/ztd/algorithm.h>
#include <zeno/ztd/variant.h>
#include <zeno/math/vec.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace zty {


namespace {


namespace op1 {

#define _OP(name, ...) \
    struct name { \
        auto operator()(auto const &a) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(ident, a)
    _OP(neg, -a)
    _OP(abs, math::abs(a))
    _OP(floor, math::floor(a))
    _OP(ceil, math::ceil(a))
    _OP(sqrt, math::sqrt(a))
    _OP(exp, math::exp(a))
    _OP(log, math::log(a))
    _OP(sin, math::sin(a))
    _OP(cos, math::cos(a))
    _OP(tan, math::tan(a))
    _OP(asin, math::asin(a))
    _OP(acos, math::acos(a))
    _OP(atan, math::atan(a))

#undef _OP

    using variant = std::variant
        < ident
        , neg
        , abs
        , floor
        , ceil
        , sqrt
        , exp
        , log
        , sin
        , cos
        , tan
        , asin
        , acos
        , atan
        >;

    static constexpr const char *type_list[] = {
        "ident",
        "neg",
        "abs",
        "floor",
        "ceil",
        "sqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}


namespace op2 {

#define _OP(name, ...) \
    struct name { \
        auto operator()(auto const &a, auto const &b) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(identl, a)
    _OP(identr, b)
    _OP(add, a + b)
    _OP(sub, a - b)
    _OP(mul, a * b)
    _OP(div, a / b)
    _OP(pow, math::pow(a, b))
    _OP(atan2, math::atan2(a, b))
    _OP(min, math::min(a, b))
    _OP(max, math::max(a, b))
    _OP(fmod, math::fmod(a, b))

#undef _OP

    using variant = std::variant
        < identl
        , identr
        , add
        , pow
        , atan2
        , min
        , max
        , fmod
        >;

    static constexpr const char *type_list[] = {
        "identl",
        "identr",
        "add",
        "pow",
        "atan2",
        "min",
        "max",
        "fmod",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}


namespace op3 {

#define _OP(name, ...) \
    struct name { \
        auto operator()(auto const &a, auto const &b, auto const &c) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(identl, a)
    _OP(identm, b)
    _OP(identr, c)
    _OP(clamp, math::clamp(a, b, c))
    _OP(lerp, math::lerp(a, b, c))
    _OP(fma, math::fma(a, b, c))

#undef _OP

    using variant = std::variant
        < identl
        , identm
        , identr
        , clamp
        , lerp
        , fma
        >;

    static constexpr const char *type_list[] = {
        "identl",
        "identm",
        "identr",
        "clamp",
        "lerp",
        "fma",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}


}


Array arrayMathOp(std::string const &type, Array const &arr1) {
    auto op = op1::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1) -> Array {
            size_t n = arr1.size();
            std::vector<decltype(op(arr1[0]))> arr(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                arr[i] = op(arr1[i]);
            }
            return arr;
        }, arr1);
    }, op);
}


Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2) {
    auto op = op2::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2) -> Array {
            size_t n = std::min(arr1.size(), arr2.size());
            std::vector<decltype(op(arr1[0], arr2[0]))> arr(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                arr[i] = op(arr1[std::min(i, n - 1)], arr2[std::min(i, n - 1)]);
            }
            return arr;
        }, arr1, arr2);
    }, op);
}


Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2, Array const &arr3) {
    auto op = op3::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2, auto const &arr3) -> Array {
            size_t n = std::min({arr1.size(), arr2.size(), arr3.size()});
            std::vector<decltype(op(arr1[0], arr2[0], arr3[0]))> arr(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                arr[i] = op(arr1[std::min(i, n - 1)], arr2[std::min(i, n - 1)], arr3[std::min(i, n - 1)]);
            }
            return arr;
        }, arr1, arr2, arr3);
    }, op);
}


}
ZENO_NAMESPACE_END
