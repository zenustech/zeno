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
        bool operator()(auto const &a, auto const &b) const { \
            if constexpr (requires { \
                bool{(__VA_ARGS__)}; \
            }) { \
                return __VA_ARGS__; \
            } else { \
                return {}; \
            } \
        } \
    };

    _OP(identl, (bool)a)
    _OP(identr, (bool)b)
    _OP(cmpeq, a == b)
    _OP(cmpne, a != b)
    _OP(cmpge, a >= b)
    _OP(cmple, a <= b)
    _OP(cmpgt, a > b)
    _OP(cmplt, a < b)

#undef _OP

    using variant = std::variant
        < identl
        , identr
        , cmpeq
        , cmpne
        , cmpge
        , cmple
        , cmpgt
        , cmplt
        >;

    static constexpr const char *type_list[] = {
        "identl",
        "identr",
        "cmpeq",
        "cmpne",
        "cmpge",
        "cmple",
        "cmpgt",
        "cmplt",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}
}


BoolArray arrayBoolOp(std::string const &type, Array const &arr1, Array const &arr2) {
    auto op = op::from_string(type);
    return std::visit([&] (auto const &op) {
        return std::visit([&] (auto const &arr1, auto const &arr2) -> BoolArray {
            BoolArray arr(arr1.size() && arr2.size() ? std::max(arr1.size(), arr2.size()) : 0);
            #pragma omp parallel for
            for (size_t t = 0; t < arr.size(); t += 64) {
                for (size_t i = t; i < std::min(arr.size(), t + 64); i++) {
                    arr[i] = op(arr1[std::min(i, arr1.size() - 1)], arr2[std::min(i, arr2.size() - 1)]);
                }
            }
            return arr;
        }, arr1.get(), arr2.get());
    }, op);
}


}
ZENO_NAMESPACE_END
