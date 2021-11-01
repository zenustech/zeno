#pragma once


#include "vec_functions.h"


ZENO_NAMESPACE_BEGIN
namespace math {


template <size_t N, class T1, class T2>
    requires (N >= 1 && requires (T1 t1, T2 t2) { t1 * t2; })
constexpr auto dot(vec<N, T1> const &t1, vec<N, T2> const &t2) {
    auto ret = t1[0] * t2[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t2[i];
    }
    return ret;
}


template <size_t N, class T1>
    requires (N >= 1 && requires (T1 t1) { std::sqrt(t1 * t1); })
constexpr auto length(vec<N, T1> const &t1) {
    auto ret = t1[0] * t1[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t1[i];
    }
    return sqrt(ret);
}


}
ZENO_NAMESPACE_END
