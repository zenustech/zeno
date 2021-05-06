// vim: sw=2 sts=2 ts=2
#pragma once


#include <cstddef>
#include <cmath>


template <class T>
struct Vec3 {
  T x, y, z;

  Vec3() = default;
  Vec3(T const &x) : Vec3(x, x, x) {}
  Vec3(T const &x, T const &y, T const &z) : x(x), y(y), z(z) {}
  Vec3(Vec3 const &o) : Vec3(o.x, o.y, o.z) {}

  template <class S>
  operator Vec3<S>() const {
    return Vec3<S>(x, y, z);
  }

  T *data() const {
    return &x;
  }

  size_t size() const {
    return 3;
  }

  T &operator[](int i) {
    return data()[i];
  }

  T const &operator[](int i) const {
    return data()[i];
  }
};


#define _PER_OP2(op) \
template <class T, class S> \
inline Vec3<T> operator op(Vec3<T> const &lhs, Vec3<S> const &rhs) { \
  return {lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z}; \
} \
\
template <class T, class S> \
inline Vec3<T> operator op(T const &lhs, Vec3<S> const &rhs) { \
  return {lhs op rhs.x, lhs op rhs.y, lhs op rhs.z}; \
} \
\
template <class T, class S> \
inline Vec3<T> operator op(Vec3<T> const &lhs, S const &rhs) { \
  return {lhs.x op rhs, lhs.y op rhs, lhs.z op rhs}; \
} \
\
template <class T, class S> \
inline Vec3<T> &operator op##=(Vec3<T> &lhs, S const &rhs) { \
  lhs = lhs op rhs; \
  return lhs; \
}
_PER_OP2(+)
_PER_OP2(-)
_PER_OP2(*)
_PER_OP2(/)
_PER_OP2(%)
_PER_OP2(&)
_PER_OP2(|)
_PER_OP2(^)
_PER_OP2(>>)
_PER_OP2(<<)
#undef _PER_OP2

#define _PER_OP1(op) \
template <class T> \
inline Vec3<T> operator op(Vec3<T> const &lhs) { \
  return {op lhs.x, op lhs.y, op lhs.z}; \
}
_PER_OP1(+)
_PER_OP1(-)
_PER_OP1(~)
_PER_OP1(!)
#undef _PER_OP1


template <class T, class F>
inline auto vapply(F const &f, Vec3<T> const &lhs) -> decltype(auto) {
  return Vec3{f(lhs.x), f(lhs.y), f(lhs.z)};
}

template <class T, class F>
inline auto vapply(F const &f, T const &lhs) -> decltype(auto) {
  return f(lhs);
}


template <class T, class S, class F>
inline auto vapply(F const &f, Vec3<T> const &lhs, Vec3<S> const &rhs) -> decltype(auto) {
  return Vec3{f(lhs.x, rhs.x), f(lhs.y, rhs.y), f(lhs.z, rhs.z)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, T const &lhs, Vec3<S> const &rhs) -> decltype(auto) {
  return Vec3{f(lhs, rhs.x), f(lhs, rhs.y), f(lhs, rhs.z)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, Vec3<T> const &lhs, S const &rhs) -> decltype(auto) {
  return Vec3{f(lhs.x, rhs), f(lhs.y, rhs), f(lhs.z, rhs)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, T const &lhs, S const &rhs) -> decltype(auto) {
  return f(lhs, rhs);
}


#define _PER_FN2(func) \
template <class T, class S> \
inline auto func(T const &lhs, S const &rhs) -> decltype(auto) { \
  return vapply([] (auto const &x, auto const &y) { return std::func(x, y); }, lhs, rhs); \
}
_PER_FN2(atan2)
_PER_FN2(pow)
#undef _PER_FN2

#define _PER_FN1(func) \
template <class T> \
inline auto func(T const &lhs) -> decltype(auto) { \
  return vapply([] (auto const &x) { return std::func(x); }, lhs); \
}
_PER_FN1(sqrt)
_PER_FN1(sin)
_PER_FN1(cos)
_PER_FN1(tan)
_PER_FN1(asin)
_PER_FN1(acos)
_PER_FN1(atan)
_PER_FN1(exp)
_PER_FN1(log)


using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;
