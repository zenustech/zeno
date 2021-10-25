#pragma once


#include "vec_type.h"
#include "vec_traits.h"
#include "concepts.h"


namespace zeno::ztd {
inline namespace math {


template <is_not_vec T1, is_not_vec T2, class Func>
constexpr auto vec_wise(T1 const &t1, T2 const &t2, Func &&func) {
    return func(t1, t2);
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
constexpr auto vec_wise(vec<N, T1> const &t1, T2 const &t2, Func &&func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1[i], t2);
    }
    return t0;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
constexpr auto vec_wise(T1 const &t1, vec<N, T2> const &t2, Func &&func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1, t2[i]);
    }
    return t0;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
constexpr auto vec_wise(vec<N, T1> const &t1, vec<N, T2> const &t2, Func &&func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1[i], t2[i]);
    }
    return t0;
}

template <is_not_vec T1, is_not_vec T2, class Func>
constexpr auto &vec_wise_assign(T1 &t1, T2 const &t2, Func &&func) {
    func(t1, t2);
    return t1;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
constexpr auto &vec_wise_assign(vec<N, T1> &t1, T2 const &t2, Func &&func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    for (size_t i = 0; i < N; i++) {
        func(t1[i], t2);
    }
    return t1;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
constexpr auto &vec_wise_assign(vec<N, T1> &t1, vec<N, T2> const &t2, Func &&func) {
    for (size_t i = 0; i < N; i++) {
        func(t1[i], t2[i]);
    }
    return t1;
}

template <class T1, class T2>
    requires (concepts::has_plus<remove_vec_t<T1>, remove_vec_t<T2>>)
constexpr decltype(auto) operator+(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 + t2; });
}

template <class T1, class T2>
    requires (concepts::has_minus<remove_vec_t<T1>, remove_vec_t<T2>>)
constexpr decltype(auto) operator-(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 - t2; });
}

template <class T1, class T2>
    requires (concepts::has_times<remove_vec_t<T1>, remove_vec_t<T2>>)
constexpr decltype(auto) operator*(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 * t2; });
}

template <class T1, class T2>
    requires (concepts::has_divide<remove_vec_t<T1>, remove_vec_t<T2>>)
constexpr decltype(auto) operator/(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 / t2; });
}

template <class T1, class T2>
    requires (concepts::has_modolus<remove_vec_t<T1>, remove_vec_t<T2>>)
constexpr decltype(auto) operator%(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 % t2; });
}

template <class T1, class T2>
    requires (concepts::has_plus_assign<remove_vec_t<T1> &, remove_vec_t<T2>>)
constexpr decltype(auto) operator+=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 += t2; });
}

template <class T1, class T2>
    requires (concepts::has_minus_assign<remove_vec_t<T1> &, remove_vec_t<T2>>)
constexpr decltype(auto) operator-=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 -= t2; });
}

template <class T1, class T2>
    requires (concepts::has_times_assign<remove_vec_t<T1> &, remove_vec_t<T2>>)
constexpr decltype(auto) operator*=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 *= t2; });
}

template <class T1, class T2>
    requires (concepts::has_divide_assign<remove_vec_t<T1> &, remove_vec_t<T2>>)
constexpr decltype(auto) operator/=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 /= t2; });
}

template <class T1, class T2>
    requires (concepts::has_modolus_assign<remove_vec_t<T1> &, remove_vec_t<T2>>)
constexpr decltype(auto) operator%=(T1 &t1, T2 const &t2) {
    return vec_wise_assign(t1, t2, [] (auto &&t1, auto &&t2) { t1 %= t2; });
}


}
}
