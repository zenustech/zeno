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
}


Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2) {
    auto op = op::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2) -> Array {
            std::vector<decltype(op(arr1[0], arr2[0]))> arr(
                arr1.size() && arr2.size() ? std::max(arr1.size(), arr2.size()) : 0);
            #pragma omp parallel for
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = op(arr1[std::min(i, arr1.size() - 1)], arr2[std::min(i, arr2.size() - 1)]);
            }
            return arr;
        }, arr1, arr2);
    }, op);
}


}
ZENO_NAMESPACE_END
