#include <zeno/zty/array/Array.h>
#include <zeno/ztd/algorithm.h>
#include <zeno/ztd/variant.h>
#include <zeno/math/vec.h>
#include <functional>


ZENO_NAMESPACE_BEGIN
namespace zty {


namespace {
namespace op {

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
}


Array arrayMathOp(std::string const &type, Array const &arr1) {
    auto op = op::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1) -> Array {
            size_t n = arr1.size();
            std::vector<decltype(op(arr1[0]))> arr(arr1.size());
            #pragma omp parallel for
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = op(arr1[i]);
            }
            return arr;
        }, arr1.get());
    }, op);
}


}
ZENO_NAMESPACE_END
