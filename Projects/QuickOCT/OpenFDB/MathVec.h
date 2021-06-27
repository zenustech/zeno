// vim: sw=2 sts=2 ts=2
#pragma once


#include <cstddef>
#include <cmath>


namespace fdb {


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
inline auto operator op(Vec3<T> const &a, Vec3<S> const &b) { \
  return Vec3{a.x op b.x, a.y op b.y, a.z op b.z}; \
} \
\
template <class T, class S> \
inline auto operator op(T const &a, Vec3<S> const &b) { \
  return Vec3{a op b.x, a op b.y, a op b.z}; \
} \
\
template <class T, class S> \
inline auto operator op(Vec3<T> const &a, S const &b) { \
  return Vec3{a.x op b, a.y op b, a.z op b}; \
}
#define _PER_IOP2(op) \
_PER_OP2(op) \
template <class T, class S> \
inline Vec3<T> &operator op##=(Vec3<T> &a, S const &b) { \
  a = a op b; \
  return a; \
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
#undef _PER_OP2

#define _PER_OP1(op) \
template <class T> \
inline Vec3<T> operator op(Vec3<T> const &a) { \
  return {op a.x, op a.y, op a.z}; \
}
_PER_OP1(+)
_PER_OP1(-)
_PER_OP1(~)
_PER_OP1(!)
#undef _PER_OP1


template <class T, class F>
inline auto vapply(F const &f, Vec3<T> const &a) {
  return Vec3{f(a.x), f(a.y), f(a.z)};
}

template <class T, class F>
inline auto vapply(F const &f, T const &a) {
  return f(a);
}


template <class T, class S, class F>
inline auto vapply(F const &f, Vec3<T> const &a, Vec3<S> const &b) {
  return Vec3{f(a.x, b.x), f(a.y, b.y), f(a.z, b.z)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, T const &a, Vec3<S> const &b) {
  return Vec3{f(a, b.x), f(a, b.y), f(a, b.z)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, Vec3<T> const &a, S const &b) {
  return Vec3{f(a.x, b), f(a.y, b), f(a.z, b)};
}

template <class T, class S, class F>
inline auto vapply(F const &f, T const &a, S const &b) {
  return f(a, b);
}


#define _PER_FN2(func) \
template <class T, class S> \
inline auto func(T const &a, S const &b) -> decltype(auto) { \
  return vapply([] (auto const &x, auto const &y) { return std::func(x, y); }, a, b); \
}
_PER_FN2(atan2)
_PER_FN2(pow)
_PER_FN2(max)
_PER_FN2(min)
#undef _PER_FN2

#define _PER_FN1(func) \
template <class T> \
inline auto func(T const &a) { \
  return vapply([] (auto const &x) { return std::func(x); }, a); \
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


template <class T>
inline auto sum(T const &a) {
  return a.x + a.y + a.z;
}

template <class T, class S>
inline auto dot(T const &a, S const &b) {
  return sum(a * b);
}

template <class T>
inline auto length(T const &a) {
  return sqrt(sum(a * a));
}

template <class T>
inline auto normalize(T const &a) {
  return a * (1 / length(a));
}

template <class T>
inline bool any(T const &a) {
  return a.x || a.y || a.z;
}

template <class T>
inline bool all(T const &a) {
  return a.x && a.y && a.z;
}


using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;
using Vec3l = Vec3<long>;
using Vec3h = Vec3<short>;
using Vec3b = Vec3<char>;
using Vec3I = Vec3<unsigned int>;
using Vec3L = Vec3<unsigned long>;
using Vec3H = Vec3<unsigned short>;
using Vec3B = Vec3<unsigned char>;


}
