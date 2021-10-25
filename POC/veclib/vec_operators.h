#pragma once


#include "vec_type.h"
#include "concepts.h"


namespace zeno::ztd {
inline namespace math {


template <class T>
struct vec_traits : std::false_type {
    static constexpr size_t dim = 0;
    using type = T;
};

template <size_t N, class T>
struct vec_traits<vec<N, T>> : std::true_type {
    static constexpr size_t dim = N;
    using type = T;

    template <class S>
    constexpr bool operator==(vec_traits<S> const &that) const {
        return !value || !that.value || dim == that.dim;
    }
};

template <class T>
using remove_vec_t = typename vec_traits<T>::type;

template <class T>
static constexpr size_t vec_dimension_v = vec_traits<T>::dim;

template <class T>
concept is_vec = vec_traits<T>::value;

template <class T>
concept is_not_vec = !is_vec<T>;


template <is_not_vec T1, is_not_vec T2, class Func>
auto vec_wise(T1 const &t1, T2 const &t2, Func func) {
    return func(t1, t2);
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
auto vec_wise(vec<N, T1> const &t1, T2 const &t2, Func func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1[i], t2);
    }
    return t0;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
auto vec_wise(T1 const &t1, vec<N, T2> const &t2, Func func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1, t2[i]);
    }
    return t0;
}

template <size_t N, is_not_vec T1, is_not_vec T2, class Func>
auto vec_wise(vec<N, T1> const &t1, vec<N, T2> const &t2, Func func) {
    using T0 = std::remove_cvref_t<std::invoke_result_t<Func, T1, T2>>;
    vec<N, T0> t0;
    for (size_t i = 0; i < N; i++) {
        t0[i] = func(t1[i], t2[i]);
    }
    return t0;
}

template <class T1, class T2>
    requires (concepts::has_plus<remove_vec_t<T1>, remove_vec_t<T2>>)
decltype(auto) operator+(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 + t2; });
}

template <class T1, class T2>
    requires (concepts::has_minus<remove_vec_t<T1>, remove_vec_t<T2>>)
decltype(auto) operator-(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 - t2; });
}

template <class T1, class T2>
    requires (concepts::has_times<remove_vec_t<T1>, remove_vec_t<T2>>)
decltype(auto) operator*(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 * t2; });
}

template <class T1, class T2>
    requires (concepts::has_divide<remove_vec_t<T1>, remove_vec_t<T2>>)
decltype(auto) operator/(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 * t2; });
}

template <class T1, class T2>
    requires (concepts::has_modolus<remove_vec_t<T1>, remove_vec_t<T2>>)
decltype(auto) operator%(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 * t2; });
}


}
}
