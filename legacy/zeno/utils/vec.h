#pragma once

#include <zeno/math/vec.h>


namespace zeno {

using namespace ZENO_NAMESPACE::math;

inline auto alltrue(auto x) {
    return vall(x);
}

inline auto anytrue(auto x) {
    return vany(x);
}

inline auto tofloat(auto x) {
    return vcast<float>(x);
}

inline auto toint(auto x) {
    return vcast<int>(x);
}

inline auto mix(auto x, auto y, auto z) {
    return lerp(x, y, z);
}

inline auto lengthSquared(auto x) {
    return dot(x, x);
}

template <class T>
static constexpr auto is_vec_n = std::max((size_t)1, vec_traits<T>::dim);

template <class T>
static constexpr auto is_vec_v = vec_traits<T>::value;

template <class T>
using decay_vec_t = typename vec_traits<T>::type;

template <class T, class S> struct is_vec_promotable : std::false_type {
  using type = void;
};

template <class T, size_t N>
struct is_vec_promotable<vec<N, T>, vec<N, T>> : std::true_type {
  using type = vec<N, T>;
};

template <class T, size_t N>
struct is_vec_promotable<vec<N, T>, T> : std::true_type {
  using type = vec<N, T>;
};

template <class T, size_t N>
struct is_vec_promotable<T, vec<N, T>> : std::true_type {
  using type = vec<N, T>;
};

template <class T> struct is_vec_promotable<T, T> : std::true_type {
  using type = T;
};

template <class T, class S>
inline constexpr bool is_vec_promotable_v =
    is_vec_promotable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T, class S>
using is_vec_promotable_t =
    typename is_vec_promotable<std::decay_t<T>, std::decay_t<S>>::type;

template <class T, class S> struct is_vec_castable : std::false_type {};
template <class T, size_t N>
struct is_vec_castable<vec<N, T>, T> : std::true_type {};

template <class T, size_t N>
struct is_vec_castable<T, vec<N, T>> : std::false_type {};

template <class T> struct is_vec_castable<T, T> : std::true_type {};

template <class T, class S>
inline constexpr bool is_vec_castable_v =
    is_vec_castable<std::decay_t<T>, std::decay_t<S>>::value;

}
