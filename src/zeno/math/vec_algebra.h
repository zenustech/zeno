#pragma once


#include "vec_functions.h"


ZENO_NAMESPACE_BEGIN
namespace math {


template <size_t N, class T1, class T2>
    requires (requires (T1 t1, T2 t2) { t1 * t2; })
constexpr auto dot(vec<N, T1> const &t1, vec<N, T2> const &t2) {
    auto ret = t1[0] * t2[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t2[i];
    }
    return ret;
}


template <size_t N, class T1>
    requires (requires (T1 t1) { sqrt(t1 * t1); })
constexpr auto length(vec<N, T1> const &t1) {
    auto ret = t1[0] * t1[0];
    for (int i = 1; i < N; i++) {
        ret += t1[i] * t1[i];
    }
    return sqrt(ret);
}


template <size_t N, class T1>
    requires (requires (T1 t1) { t1 / sqrt(t1 * t1); })
constexpr auto normalize(vec<N, T1> const &t1) {
    return t1 / length(t1);
}


template <size_t N, class T1, class T2>
    requires (requires (T1 t1, T2 t2) { sqrt((t1 - t2) * (t1 - t2)); })
constexpr auto distance(vec<N, T1> const &t1, vec<N, T2> const &t2) {
    return length(t1 - t2);
}


template <class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto cross(vec<2, T1> const &t1, vec<2, T2> const &t2) {
    return t1[0] * t2[1] - t2[0] * t1[1];
}


template <class T1, class T2>
    requires (!vec_traits<T1>::value && !vec_traits<T2>::value)
constexpr auto cross(vec<3, T1> const &t1, vec<3, T2> const &t2) {
    return vec<3, std::remove_cvref_t<decltype(t1[0] * t2[0])>>(
      t1[1] * t2[2] - t2[1] * t1[2],
      t1[2] * t2[0] - t2[2] * t1[0],
      t1[0] * t2[1] - t2[0] * t1[1]);
}


}
ZENO_NAMESPACE_END
