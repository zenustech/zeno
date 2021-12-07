#pragma once


#include <vector>
#include <string>
#include <variant>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


using Array = std::variant
    < std::vector<float>
    , std::vector<int>
    , std::vector<math::vec3f>
    >;


template <class T>
inline std::vector<T> &arrayGet(Array &arr) {
    return std::get<std::vector<T>>(arr);
}

template <class T>
inline std::vector<T> const &arrayGet(Array const &arr) {
    return std::get<std::vector<T>>(arr);
}

template <class T>
inline std::vector<T> arrayGet(Array &&arr) {
    return std::get<std::vector<T>>(std::move(arr));
}


Array arrayMathOp(std::string const &type, Array const &arr1);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2);
Array arrayMathOp(std::string const &type, Array const &arr1, Array const &arr2, Array const &arr3);


}
ZENO_NAMESPACE_END
