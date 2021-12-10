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
        bool operator()(bool a, bool b) const { \
            return __VA_ARGS__; \
        } \
    };

    _OP(identl, a)
    _OP(identr, b)
    _OP(logicnotl, !a)
    _OP(logicnotr, !b)
    _OP(logicand, a && b)
    _OP(logicor, a || b)
    _OP(logicxor, a != b)
    _OP(logicnor, a == b)

#undef _OP

    using variant = std::variant
        < identl
        , identr
        , logicnotl
        , logicnotr
        , logicand
        , logicor
        , logicxor
        , logicnor
        >;

    static constexpr const char *type_list[] = {
        "identl",
        "identr",
        "logicnotl",
        "logicnotr",
        "logicand",
        "logicor",
        "logicxor",
        "logicnor",
    };

    variant from_string(std::string const &type) {
        size_t index = ztd::find_index(type_list, type);
        return ztd::variant_from_index<variant>(index);
    }
}
}


BoolArray arrayBoolOp(std::string const &type, BoolArray const &arr1, BoolArray const &arr2) {
    auto op = op::from_string(type);
    return std::visit([&] (auto const &op) -> BoolArray {
        BoolArray arr(arr1.size() && arr2.size() ? std::max(arr1.size(), arr2.size()) : 0);
        #pragma omp parallel for
        for (size_t t = 0; t < arr.size(); t += 64) {
            for (size_t i = t; i < std::min(arr.size(), t + 64); i++) {
                arr[i] = op(arr1[std::min(i, arr1.size() - 1)], arr2[std::min(i, arr2.size() - 1)]);
            }
        }
        return arr;
    }, op);
}


BoolArray arrayBoolNotOp(BoolArray const &arr1) {
    return arrayBoolOp("logicnotl", arr1, BoolArray{false});
}


Array arrayFromBoolOp(BoolArray const &arr1) {
    std::vector<int> arr(arr1.size());
    #pragma omp parallel for
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = arr1[i] ? 1 : 0;
    }
    return arr;
}


}
ZENO_NAMESPACE_END
