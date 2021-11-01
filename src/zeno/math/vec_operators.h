#pragma once


#include "vec_type.h"
#include "vec_traits.h"
#include "vec_elmwise.h"


ZENO_NAMESPACE_BEGIN
namespace math {

template <class T1>
    requires (is_vec<T1> && requires (remove_vec_t<T1> t1) { +t1; })
constexpr decltype(auto) operator+(T1 const &t1) {
    return vec_wise(t1, [] (auto &&t1) { return +t1; });
}

template <class T1>
    requires (is_vec<T1> && requires (remove_vec_t<T1> t1) { +t1; })
constexpr decltype(auto) operator-(T1 const &t1) {
    return vec_wise(t1, [] (auto &&t1) { return -t1; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 + t2; })
constexpr decltype(auto) operator+(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 + t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 - t2; })
constexpr decltype(auto) operator-(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 - t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 * t2; })
constexpr decltype(auto) operator*(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 * t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 / t2; })
constexpr decltype(auto) operator/(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 / t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 % t2; })
constexpr decltype(auto) operator%(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 % t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> &t1, remove_vec_t<T2> t2) { t1 += t2; })
constexpr decltype(auto) operator+=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 += t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> &t1, remove_vec_t<T2> t2) { t1 -= t2; })
constexpr decltype(auto) operator-=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 -= t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> &t1, remove_vec_t<T2> t2) { t1 *= t2; })
constexpr decltype(auto) operator*=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 *= t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> &t1, remove_vec_t<T2> t2) { t1 /= t2; })
constexpr decltype(auto) operator/=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 /= t2; });
}

template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> &t1, remove_vec_t<T2> t2) { t1 %= t2; })
constexpr decltype(auto) operator%=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 %= t2; });
}


}
ZENO_NAMESPACE_END
