#pragma once


#include "vec_functions.h"


ZENO_NAMESPACE_BEGIN
namespace math {


template <size_t N, class T1, class T2>
    requires (N >= 1 && requires (T1 t1, T2 t2) { t1 * t2; })
constexpr auto dot(vec<T1, N> const &t1, vec<T2, N> const &t2) {
    auto ret = t1[0] * t2[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t2[i];
    }
    return ret;
}


template <size_t N, class T1, class T2>
    requires (N >= 1 && requires (T1 t1, T2 t2) { std::sqrt(t1 * t2); })
constexpr auto length(vec<T1, N> const &t1, vec<T2, N> const &t2) {
    auto ret = t1[0] * t2[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t2[i];
    }
    return sqrt(ret);
}


}
