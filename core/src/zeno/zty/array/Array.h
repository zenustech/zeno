#pragma once


#include <vector>
#include <variant>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


using Array = std::variant
    < std::vector<float>
    , std::vector<uint32_t>
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


auto arrayVectorApply(auto const &func, auto &&...arrs) {
    return std::visit([&] (auto &&...arrs) {
        func(arrs...);
    }, std::forward<decltype(arrs)>(arrs)...);
}


void arraySerialApply(auto const &func, auto &&...arrs) {
    arrayVectorApply([&] (auto &&...arrs) {
        for (size_t i = 0; i < std::max({arrs.size()...}); i++) {
            func(arrs[i % arrs.size()]...);
        }
    }, std::forward<decltype(arrs)>(arrs)...);
}


void arrayParallelApply(auto const &func, auto &&...arrs) {
    arrayVectorApply([&] (auto &&...arrs) {
#pragma omp parallel for
        for (size_t i = 0; i < std::max({arrs.size()...}); i++) {
            func(arrs[i % arrs.size()]...);
        }
    }, std::forward<decltype(arrs)>(arrs)...);
}



}
ZENO_NAMESPACE_END
