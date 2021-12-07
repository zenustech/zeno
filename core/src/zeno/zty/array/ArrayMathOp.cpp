#include <zeno/zty/array/Array.h>
#include <zeno/ztd/algorithm.h>
#include <zeno/ztd/variant.h>
#include <zeno/math/vec.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace zty {


namespace op1 {

#define _OP(name, ...) \
    struct name { \
        decltype(auto) operator()(auto &&a) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(ident, a)
    _OP(neg, -a)
    _OP(sin, math::sin(a))
    _OP(cos, math::cos(a))
    _OP(tan, math::tan(a))

#undef _OP

    using variant = std::variant
        < ident
        , neg
        , sin
        , cos
        , tan
        >;

    static constexpr const char *type_list[] = {
        "ident",
        "neg",
        "sin",
        "cos",
        "tan",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}


namespace op2 {

#define _OP(name, ...) \
    struct name { \
        decltype(auto) operator()(auto &&a, auto &&b) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(identl, a)
    _OP(identr, b)

#undef _OP

    using variant = std::variant
        < identl
        , identr
        >;

    static constexpr const char *type_list[] = {
        "identl",
        "identr",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}


Array arrayMathOp(std::string const &type, Array const &arr1) {
    auto op = op1::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1) -> Array {
            auto n = arr1.size();
            std::vector<std::decay_t<decltype(op(arr1[0]))>> arrout(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                arrout[i] = op(arr1[i]);
            }
            return arrout;
        }, arr1);
    }, op);
}


Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2) {
    auto op = op2::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2) -> Array {
            auto n = std::min(arr1.size(), arr2.size());
            std::vector<std::decay_t<decltype(op(arr1[0], arr2[0]))>> arrout(n);
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                arrout[i] = op(arr1[std::min(i, n - 1)], arr2[std::min(i, n - 1)]);
            }
            return arrout;
        }, arr1, arr2);
    }, op);
}


}
ZENO_NAMESPACE_END
