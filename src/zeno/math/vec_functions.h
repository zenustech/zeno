#pragma once


#include "vec_operators.h"
#include <cmath>


ZENO_NAMESPACE_BEGIN
namespace math {

#define _OP3(fname) \
template <class T1, class T2, class T3, \
          class T0 = decltype(std::declval<remove_vec_t<T1>>() \
                            + std::declval<remove_vec_t<T2>>() + std::declval<remove_vec_t<T3>>()> \
    requires (requires (T0 t0) { std::fname(t0, t0, t0); }) \
constexpr decltype(auto) fname(T1 const &t1, T2 const &t2, T3 const &t3) { \
    return vec_wise(t1, t2, t3, [] (auto &&t1, auto &&t2, auto &&t3) { return (T0)std::fname((T0)t1, (T0)t2, (T0)t3); }); \
}

#define _OP2(fname) \
template <class T1, class T2, \
          class T0 = decltype(std::declval<remove_vec_t<T1>>() + std::declval<remove_vec_t<T2>>())> \
    requires (requires (T0 t0) { std::fname(t0, t0); }) \
constexpr decltype(auto) fname(T1 const &t1, T2 const &t2) { \
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return (T0)std::fname((T0)t1, (T0)t2); }); \
}

#define _OP1(fname) \
template <class T1> \
    requires (requires (T1 t1) { std::fname(t1); }) \
constexpr decltype(auto) fname(T1 const &t1) { \
    return vec_wise(t1, [] (auto &&t1) { return (T1)std::fname(t1); }); \
}

_OP3(fma)
_OP3(lerp)
_OP3(clamp)

_OP2(max)
_OP2(min)
_OP2(fmax)
_OP2(fmin)
_OP2(fmod)
_OP2(atan2)
_OP2(pow)

_OP1(sqrt)
_OP1(log)
_OP1(exp)
_OP1(abs)
_OP1(fabs)
_OP1(floor)
_OP1(ceil)

_OP1(sin)
_OP1(cos)
_OP1(tan)
_OP1(asin)
_OP1(acos)
_OP1(atan)
_OP1(sinh)
_OP1(cosh)
_OP1(tanh)
_OP1(asinh)
_OP1(acosh)
_OP1(atanh)


}
ZENO_NAMESPACE_END
