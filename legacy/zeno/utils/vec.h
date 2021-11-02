#pragma once

#include <zeno/math/vec.h>


namespace zeno {

using namespace ZENO_NAMESPACE::math;

template <size_t N, class T>
struct vec : ZENO_NAMESPACE::math::vec<N, T> {
    using ZENO_NAMESPACE::math::vec<N, T>::vec;
};

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

template <class T1, class T2>
    requires (is_vec_v<T1> || is_vec_v<T2>)
constexpr auto operator!=(T1 const &t1, T2 const &t2) {
    return vcmpne(t1, t2);
}

template <class T1, class T2>
    requires (is_vec_v<T1> || is_vec_v<T2>)
constexpr auto operator>=(T1 const &t1, T2 const &t2) {
    return vcmpge(t1, t2);
}

template <class T1, class T2>
    requires (is_vec_v<T1> || is_vec_v<T2>)
constexpr auto operator>(T1 const &t1, T2 const &t2) {
    return vcmpgt(t1, t2);
}

template <class T1, class T2>
    requires (is_vec_v<T1> || is_vec_v<T2>)
constexpr auto operator<=(T1 const &t1, T2 const &t2) {
    return vcmple(t1, t2);
}

template <class T1, class T2>
    requires (is_vec_v<T1> || is_vec_v<T2>)
constexpr auto operator<(T1 const &t1, T2 const &t2) {
    return vcmplt(t1, t2);
}

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
