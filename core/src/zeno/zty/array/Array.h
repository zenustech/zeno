#pragma once


#include <vector>
#include <variant>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct Array {
    std::variant
    < std::vector<float>
    , std::vector<uint32_t>
    , std::vector<math::vec3f>
    > arr;

    template <class T>
    inline std::vector<T> &get() {
        return std::get<std::vector<T>>(arr);
    }

    template <class T>
    inline std::vector<T> const &get() const {
        return std::get<std::vector<T>>(arr);
    }
};


auto arrayVectorApply(auto const &func, auto &&...arrs) {
    return std::visit([&] (auto &&...arrs) {
        func(arrs.arr...);
    }, std::forward<decltype(arrs)>(arrs.arr)...);
}


void arraySerialApply(auto const &func, auto &&...arrs) {
    arrayVisit([&] (auto &&...arrs) {
        for (size_t i = 0; i < std::max({arrs.size()...}); i++) {
            func(arrs[i % arrs.size()]...);
        }
    }, std::forward<decltype(arrs)>(arrs)...);
}


void arrayParallelApply(auto const &func, auto &&...arrs) {
    arrayVisit([&] (auto &&...arrs) {
#pragma omp parallel for
        for (size_t i = 0; i < std::max({arrs.size()...}); i++) {
            func(arrs[i % arrs.size()]...);
        }
    }, std::forward<decltype(arrs)>(arrs)...);
}



}
ZENO_NAMESPACE_END
