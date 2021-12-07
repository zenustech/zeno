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


Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2, Array const &arr3) {
    auto op = op::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2, auto const &arr3) -> Array {
            std::vector<decltype(op(arr1[0], arr2[0], arr3[0]))> arr(
                arr1.size() && arr2.size() && arr3.size() ? std::max({arr1.size(), arr2.size(), arr3.size()}) : 0);
            #pragma omp parallel for
            for (size_t i = 0; i < arr.size(); i++) {
                arr[i] = op(arr1[std::min(i, arr1.size() - 1)], arr2[std::min(i, arr2.size() - 1)], arr3[std::min(i, arr3.size() - 1)]);
            }
            return arr;
        }, arr1.get(), arr2.get(), arr3.get());
    }, op);
}


}
ZENO_NAMESPACE_END
