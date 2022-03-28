#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace zeno {
template <class T, class S>
constexpr bool is_decay_same_v = std::is_same_v<std::decay_t<T>, std::decay_t<S>>;

/* main class definition */

template <size_t N, class T> struct vec : std::array<T, N> {
  vec() = default;
  explicit vec(T const &x) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] = x;
    }
  }

  vec(vec &&) = default;
  vec(vec const &) = default;
  vec &operator=(vec const &) = default;

  vec(std::array<T, N> const &a) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] = a[i];
    }
  }

  operator std::array<T, N>() const {
    std::array<T, N> res;
    for (size_t i = 0; i < N; i++) {
      res[i] = (*this)[i];
    }
    return res;
  }

  template <class S>
  explicit vec(vec<N, S> const &x) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] = T(x[i]);
    }
  }

  vec(std::initializer_list<T> const &x) {
    T val;
    auto it = x.begin();
    for (size_t i = 0; i < N; i++) {
      if (it != x.end())
        val = *it++;
      (*this)[i] = val;
    }
  }

  vec(T const &x, T const &y) : vec{x, y} {}

  vec(T const &x, T const &y, T const &z) : vec{x, y, z} {}

  vec(T const &x, T const &y, T const &z, T const &w) : vec{x, y, z, w} {}

  template <class S>
  operator vec<N, S>() const {
    vec<N, S> res;
    for (size_t i = 0; i < N; i++) {
      res[i] = (*this)[i];
    }
    return res;
  }
};

/* type traits */

template <class T>
struct is_vec : std::false_type {
  static constexpr size_t _N = 1;
};

template <size_t N, class T>
struct is_vec<vec<N, T>> : std::true_type {
  static constexpr size_t _N = N;
};

template <class T>
inline constexpr bool is_vec_v = is_vec<std::decay_t<T>>::value;

template <class T>
inline constexpr size_t is_vec_n = is_vec<std::decay_t<T>>::_N;

template <class T, class S>
struct is_vec_promotable : std::false_type {
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

template <class T>
struct is_vec_promotable<T, T> : std::true_type {
  using type = T;
};

template <class T, class S>
inline constexpr bool is_vec_promotable_v =
    is_vec_promotable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T, class S>
using is_vec_promotable_t =
    typename is_vec_promotable<std::decay_t<T>, std::decay_t<S>>::type;

template <class T, class S>
struct is_vec_castable : std::false_type {};
template <class T, size_t N>
struct is_vec_castable<vec<N, T>, T> : std::true_type {};

template <class T, size_t N>
struct is_vec_castable<T, vec<N, T>> : std::false_type {};

template <class T>
struct is_vec_castable<T, T> : std::true_type {};

template <class T, class S>
inline constexpr bool is_vec_castable_v =
    is_vec_castable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T>
struct decay_vec { using type = T; };

template <size_t N, class T>
struct decay_vec<vec<N, T>> { using type = T; };

template <class T>
using decay_vec_t = typename decay_vec<T>::type;

/* converter functions */

template <class T, std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto vec_to_other(T const &a) {
  return a;
}

template <class OtherT, class T>
inline auto vec_to_other(vec<2, T> const &a) {
  return OtherT(a[0], a[1]);
}

template <class OtherT, class T>
inline auto vec_to_other(vec<3, T> const &a) {
  return OtherT(a[0], a[1], a[2]);
}

template <class OtherT, class T>
inline auto vec_to_other(vec<4, T> const &a) {
  return OtherT(a[0], a[1], a[2], a[3]);
}

template <class OtherT, size_t N, class T>
inline auto vec_to_other(vec<N, T> const &a) {
  OtherT res;
  for (size_t i = 0; i < N; i++) {
    res[i] = a[i];
  }
  return res;
}

template <size_t N, class OtherT>
inline auto other_to_vec(OtherT const &x) {
  vec<N, std::decay_t<decltype(x[0])>> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = x[i];
  }
  return res;
}

/* element-wise operations */

template <size_t N, class T, class F>
inline auto vapply(F const &f, vec<N, T> const &a) {
  vec<N, decltype(f(a[0]))> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a[i]);
  }
  return res;
}

template <class T, class F, std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto vapply(F const &f, T const &a) {
  return f(a);
}

template <size_t N, class T, class S, class F>
inline auto vapply(F const &f, vec<N, T> const &a, vec<N, S> const &b) {
  vec<N, decltype(f(a[0], b[0]))> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a[i], b[i]);
  }
  return res;
}

template <size_t N, class T, class S, class F,
          std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto vapply(F const &f, T const &a, vec<N, S> const &b) {
  vec<N, decltype(f(a, b[0]))> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a, b[i]);
  }
  return res;
}

template <size_t N, class T, class S, class F,
          std::enable_if_t<!is_vec_v<S>, bool> = true>
inline auto vapply(F const &f, vec<N, T> const &a, S const &b) {
  vec<N, decltype(f(a[0], b))> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a[i], b);
  }
  return res;
}

template <class T, class S, class F,
          std::enable_if_t<!is_vec_v<T> && !is_vec_v<S>, bool> = true>
inline auto vapply(F const &f, T const &a, S const &b) {
  return f(a, b);
}

template <class T, class S>
inline constexpr bool
    is_vapply_v = (is_vec_v<T> || is_vec_v<S>)&&(!is_vec_v<T> || !is_vec_v<S> ||
                                                 is_vec_n<T> == is_vec_n<S>);

template <class T, class S, class F,
          std::enable_if_t<is_vapply_v<T, S>, bool> = true>
inline auto vapply(F const &f, T const &a, S const &b) {
  return f(a, b);
}

#define _PER_OP2(op)                                                           \
  template <class T, class S,                                                  \
            std::enable_if_t<is_vapply_v<T, S>, bool> = true,                  \
            decltype(std::declval<decay_vec_t<T>>()                            \
                         op std::declval<decay_vec_t<S>>(),                    \
                     true) = true>                                             \
  inline auto operator op(T const &a, S const &b)->decltype(auto) {            \
    return vapply([](auto const &x, auto const &y) { return x op y; }, a, b);  \
  }
#define _PER_IOP2(op)                                                          \
  _PER_OP2(op)                                                                 \
  template <size_t N, class T, class S,                                        \
            std::enable_if_t<is_vapply_v<vec<N, T>, S>, bool> = true,          \
            decltype(std::declval<vec<N, T>>() op std::declval<S>(), true) =   \
                true>                                                          \
  inline vec<N, T> &operator op##=(vec<N, T> &a, S const &b) {                 \
    a = a op b;                                                                \
    return a;                                                                  \
  }
_PER_IOP2(+)
_PER_IOP2(-)
_PER_IOP2(*)
_PER_IOP2(/)
_PER_IOP2(%)
_PER_IOP2(&)
_PER_IOP2(|)
_PER_IOP2(^)
_PER_IOP2(>>)
_PER_IOP2(<<)
_PER_OP2(==)
_PER_OP2(!=)
_PER_OP2(<)
_PER_OP2(>)
_PER_OP2(<=)
_PER_OP2(>=)
_PER_OP2(&&)
_PER_OP2(||)
#undef _PER_IOP2
#undef _PER_OP2

#define _PER_OP1(op)                                                           \
  template <class T, std::enable_if_t<is_vec_v<T>, bool> = true,               \
            decltype(op std::declval<decay_vec_t<T>>(), true) = true>          \
  inline auto operator op(T const &a) {                                        \
    return vapply([](auto const &x) { return op x; }, a);                      \
  }
_PER_OP1(+)
_PER_OP1(-)
_PER_OP1(~)
_PER_OP1(!)
#undef _PER_OP1

#define _PER_FN2(func)                                                         \
  template <class T, class S,                                                  \
            decltype(std::declval<T>() + std::declval<S>(), true) = true>      \
  inline auto func(T const &a, S const &b)->decltype(auto) {                   \
    return vapply(                                                             \
        [](auto const &x, auto const &y) {                                     \
          using promoted = decltype(x + y);                                    \
          return (promoted)std::func((promoted)x, (promoted)y);                \
        },                                                                     \
        a, b);                                                                 \
  }
_PER_FN2(atan2)
_PER_FN2(pow)
_PER_FN2(max)
_PER_FN2(min)
_PER_FN2(fmod)
#undef _PER_FN2

#define _PER_FN1(func)                                                         \
  template <class T> inline auto func(T const &a) {                            \
    return vapply([](auto const &x) { return (decltype(x))std::func(x); }, a); \
  }
_PER_FN1(abs)
_PER_FN1(sqrt)
_PER_FN1(sin)
_PER_FN1(cos)
_PER_FN1(tan)
_PER_FN1(asin)
_PER_FN1(acos)
_PER_FN1(atan)
_PER_FN1(exp)
_PER_FN1(log)
_PER_FN1(floor)
_PER_FN1(ceil)
#undef _PER_FN1

template <class T>
inline auto fract(T const &a) {
  return a - floor(a);
}

template <class T>
inline auto ifloor(T const &a) {
  return toint(floor(a));
}

template <class To, class T> inline auto cast(T const &a) {
  return vapply([](auto const &x) { return (To)x; }, a);
}

template <class T> inline auto toint(T const &a) {
  return cast<int, T>(a);
}

template <class T> inline auto tofloat(T const &a) {
  return cast<float, T>(a);
}

/* vector math functions */

template <size_t N, class T>
inline bool anytrue(vec<N, T> const &a) {
  bool ret = false;
  for (size_t i = 0; i < N; i++) {
    ret = ret || (bool)a[i];
  }
  return ret;
}

template <class T>
inline bool anytrue(T const &a) {
    return (bool)a;
}

template <size_t N, class T>
inline bool alltrue(vec<N, T> const &a) {
  bool ret = true;
  for (size_t i = 0; i < N; i++) {
    ret = ret && (bool)a[i];
  }
  return ret;
}

template <class T>
inline bool alltrue(T const &a) {
    return (bool)a;
}

inline auto dot(float a, float b) { return a * b; }

template <size_t N, class T, class S>
inline auto dot(vec<N, T> const &a, vec<N, S> const &b) {
  std::decay_t<decltype(a[0] * b[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * b[i];
  }
  return res;
}

template <size_t N, class T>
inline auto lengthSquared(vec<N, T> const &a) {
  std::decay_t<decltype(a[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * a[i];
  }
  return res;
}

template <size_t N, class T>
inline auto length(vec<N, T> const &a) {
  std::decay_t<decltype(a[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * a[i];
  }
  return sqrt(res);
}

template <size_t N, class T>
inline auto distance(vec<N, T> const &a, vec<N, T> const &b) {
  return length(b - a);
}

template <size_t N, class T>
inline auto normalize(vec<N, T> const &a) {
  return a * (1 / length(a));
}

template <class T, class S>
inline auto cross(vec<2, T> const &a, vec<2, S> const &b) {
  return a[0] * b[1] - b[0] * a[1];
}

template <class T, class S>
inline auto cross(vec<3, T> const &a, vec<3, S> const &b) {
  return vec<3, decltype(a[0] * b[0])>(a[1] * b[2] - b[1] * a[2],
                                       a[2] * b[0] - b[2] * a[0],
                                       a[0] * b[1] - b[0] * a[1]);
}

/* generic helper functions */

template <class T, class S, class F>
inline auto mix(T const &a, S const &b, F const &f) {
  return a * (1 - f) + b * f;
}

template <class T, class S, class F>
inline auto clamp(T const &x, S const &a, F const &b) {
  return min(max(x, a), b);
}

template <class T, class S>
inline auto minmax(T const &a, S const &b) {
  return std::make_pair(min(a, b), max(a, b));
}

template <size_t N, class T, std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto tovec(T const &x) {
  return vec<N, T>(x);
}

template <size_t N, class T>
inline auto tovec(vec<N, T> const &x) { return x; }

/* common type definitions */

using vec2f = vec<2, float>;
using vec2d = vec<2, double>;
using vec2i = vec<2, int32_t>;
using vec2l = vec<2, intptr_t>;
using vec2h = vec<2, int16_t>;
using vec2c = vec<2, int8_t>;
using vec2b = vec<2, bool>;
using vec2I = vec<2, uint32_t>;
using vec2L = vec<2, uintptr_t>;
using vec2Q = vec<2, uint64_t>;
using vec2H = vec<2, uint16_t>;
using vec2C = vec<2, uint8_t>;
using vec3f = vec<3, float>;
using vec3d = vec<3, double>;
using vec3i = vec<3, int32_t>;
using vec3l = vec<3, intptr_t>;
using vec3h = vec<3, int16_t>;
using vec3c = vec<3, int8_t>;
using vec3b = vec<3, bool>;
using vec3I = vec<3, uint32_t>;
using vec3L = vec<3, uintptr_t>;
using vec3Q = vec<3, uint64_t>;
using vec3H = vec<3, uint16_t>;
using vec3C = vec<3, uint8_t>;
using vec4f = vec<4, float>;
using vec4d = vec<4, double>;
using vec4i = vec<4, int32_t>;
using vec4l = vec<4, intptr_t>;
using vec4h = vec<4, int16_t>;
using vec4c = vec<4, int8_t>;
using vec4b = vec<4, bool>;
using vec4I = vec<4, uint32_t>;
using vec4L = vec<4, uintptr_t>;
using vec4Q = vec<4, uint64_t>;
using vec4H = vec<4, uint16_t>;
using vec4C = vec<4, uint8_t>;

}

/* specialization for structual-binding */

namespace std {

template <size_t N, class T>
struct tuple_size<::zeno::vec<N, T>> : integral_constant<size_t, N> {
};

template <size_t I, size_t N, class T>
struct tuple_element<I, ::zeno::vec<N, T>> {
    using type = enable_if_t<(I < N), T>;
};

template <size_t I, size_t N, class T>
T const &get(::zeno::vec<N, T> const &t) {
    return t[I];
}

template <size_t I, size_t N, class T>
T &get(::zeno::vec<N, T> &t) {
    return t[I];
}

}
