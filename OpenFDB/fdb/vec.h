#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace fdb {

using std::size_t;

/* main class definition */

template <class T, size_t N> struct vec : std::array<T, N> {
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

  operator std::array<T, N>() {
    std::array<T, N> res;
    for (size_t i = 0; i < N; i++) {
      res[i] = (*this)[i];
    }
    return res;
  }

  template <class S>
  explicit vec(vec<S, N> const &x) {
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
  operator vec<S, N>() const {
    vec<S, N> res;
    for (size_t i = 0; i < N; i++) {
      res[i] = (*this)[i];
    }
    return res;
  }
};

/* type traits */

template <class T> struct is_vec : std::false_type {
  static constexpr size_t _N = 1;
};

template <size_t N, class T> struct is_vec<vec<T, N>> : std::true_type {
  static constexpr size_t _N = N;
};

template <class T>
inline constexpr bool is_vec_v = is_vec<std::decay_t<T>>::value;

template <class T>
inline constexpr size_t is_vec_n = is_vec<std::decay_t<T>>::_N;

template <class T, class S> struct is_vec_promotable : std::false_type {
  using type = void;
};

template <class T, size_t N>
struct is_vec_promotable<vec<T, N>, vec<T, N>> : std::true_type {
  using type = vec<T, N>;
};

template <class T, size_t N>
struct is_vec_promotable<vec<T, N>, T> : std::true_type {
  using type = vec<T, N>;
};

template <class T, size_t N>
struct is_vec_promotable<T, vec<T, N>> : std::true_type {
  using type = vec<T, N>;
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
struct is_vec_castable<vec<T, N>, T> : std::true_type {};

template <class T, size_t N>
struct is_vec_castable<T, vec<T, N>> : std::false_type {};

template <class T> struct is_vec_castable<T, T> : std::true_type {};

template <class T, class S>
inline constexpr bool is_vec_castable_v =
    is_vec_castable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T> struct decay_vec { using type = T; };

template <size_t N, class T> struct decay_vec<vec<T, N>> { using type = T; };

template <class T> using decay_vec_t = typename decay_vec<T>::type;

/* converter functions */

template <class T, std::enable_if_t<!is_vec_v<T>, bool> = true>
auto vec_to_other(T const &a) {
  return a;
}

template <class OtherT, class T>
auto vec_to_other(vec<T, 2> const &a) {
  return OtherT(a[0], a[1]);
}

template <class OtherT, class T>
auto vec_to_other(vec<T, 3> const &a) {
  return OtherT(a[0], a[1], a[2]);
}

template <class OtherT, class T>
auto vec_to_other(vec<T, 4> const &a) {
  return OtherT(a[0], a[1], a[2], a[3]);
}

template <class OtherT, size_t N, class T>
auto vec_to_other(vec<T, N> const &a) {
  OtherT res;
  for (size_t i = 0; i < N; i++) {
    res[i] = a[i];
  }
  return res;
}

template <size_t N, class OtherT> auto other_to_vec(OtherT const &x) {
  vec<std::decay_t<decltype(x[0])>, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = x[i];
  }
  return res;
}

/* element-wise operations */

template <size_t N, class T, class F>
inline auto vapply(F const &f, vec<T, N> const &a) {
  vec<decltype(f(a[0])), N> res;
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
inline auto vapply(F const &f, vec<T, N> const &a, vec<S, N> const &b) {
  vec<decltype(f(a[0], b[0])), N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a[i], b[i]);
  }
  return res;
}

template <size_t N, class T, class S, class F,
          std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto vapply(F const &f, T const &a, vec<S, N> const &b) {
  vec<decltype(f(a, b[0])), N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = f(a, b[i]);
  }
  return res;
}

template <size_t N, class T, class S, class F,
          std::enable_if_t<!is_vec_v<S>, bool> = true>
inline auto vapply(F const &f, vec<T, N> const &a, S const &b) {
  vec<decltype(f(a[0], b)), N> res;
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
            std::enable_if_t<is_vapply_v<vec<T, N>, S>, bool> = true,          \
            decltype(std::declval<vec<T, N>>() op std::declval<S>(), true) =   \
                true>                                                          \
  inline vec<T, N> &operator op##=(vec<T, N> &a, S const &b) {                 \
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
inline bool anyTrue(vec<T, N> const &a) {
  bool ret = false;
  for (size_t i = 0; i < N; i++) {
    ret = ret || (bool)a[i];
  }
  return ret;
}

template <class T>
inline bool anyTrue(T const &a) {
    return (bool)a;
}

template <size_t N, class T>
inline bool allTrue(vec<T, N> const &a) {
  bool ret = true;
  for (size_t i = 0; i < N; i++) {
    ret = ret && (bool)a[i];
  }
  return ret;
}

template <class T>
inline bool allTrue(T const &a) {
    return (bool)a;
}

inline auto dot(float a, float b) { return a * b; }

template <size_t N, class T, class S>
inline auto dot(vec<T, N> const &a, vec<S, N> const &b) {
  std::decay_t<decltype(a[0] * b[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * b[i];
  }
  return res;
}

template <size_t N, class T>
inline auto lengthSquared(vec<T, N> const &a) {
  std::decay_t<decltype(a[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * a[i];
  }
  return res;
}

template <size_t N, class T>
inline auto length(vec<T, N> const &a) {
  std::decay_t<decltype(a[0])> res(0);
  for (size_t i = 0; i < N; i++) {
    res += a[i] * a[i];
  }
  return std::sqrt(res);
}

template <size_t N, class T>
inline auto normalize(vec<T, N> const &a) {
  return a * (1 / length(a));
}

template <class T, class S>
inline auto cross(vec<T, 2> const &a, vec<S, 2> const &b) {
  return a[0] * b[1] - b[0] * a[1];
}

template <class T, class S>
inline auto cross(vec<T, 3> const &a, vec<S, 3> const &b) {
  return vec<decltype(a[0] * b[0]), 3>(a[1] * b[2] - b[1] * a[2],
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

template <size_t N, class T, std::enable_if_t<!is_vec_v<T>, bool> = true>
inline auto tovec(T const &x) {
  return vec<T, N>(x);
}

template <size_t N, class T>
inline auto tovec(vec<T, N> const &x) { return x; }

/* common types definitions */

using vec2f = vec<float, 2>;
using vec2d = vec<double, 2>;
using vec2i = vec<int, 2>;
using vec2l = vec<long, 2>;
using vec2h = vec<short, 2>;
using vec2b = vec<char, 2>;
using vec2I = vec<unsigned int, 2>;
using vec2L = vec<unsigned long, 2>;
using vec2H = vec<unsigned short, 2>;
using vec2B = vec<unsigned char, 2>;

using vec3f = vec<float, 3>;
using vec3d = vec<double, 3>;
using vec3i = vec<int, 3>;
using vec3l = vec<long, 3>;
using vec3h = vec<short, 3>;
using vec3b = vec<char, 3>;
using vec3I = vec<unsigned int, 3>;
using vec3L = vec<unsigned long, 3>;
using vec3H = vec<unsigned short, 3>;
using vec3B = vec<unsigned char, 3>;

using vec4f = vec<float, 4>;
using vec4d = vec<double, 4>;
using vec4i = vec<int, 4>;
using vec4l = vec<long, 4>;
using vec4h = vec<short, 4>;
using vec4b = vec<char, 4>;
using vec4I = vec<unsigned int, 4>;
using vec4L = vec<unsigned long, 4>;
using vec4H = vec<unsigned short, 4>;
using vec4B = vec<unsigned char, 4>;

}
