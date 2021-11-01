#pragma once


#include "vec_operators.h"


ZENO_NAMESPACE_BEGIN
namespace math::bvec {


template <size_t N, class T1>
    requires (requires (T1 t1) { (bool)t1; })
constexpr bool vany(vec<N, T1> const &t1) {
    bool ret = (bool)t1[0];
    for (int i = 1; i < N; i++) {
        ret = ret || (bool)t1[i];
    }
    return ret;
}


template <size_t N, class T1>
    requires (requires (T1 t1) { (bool)t1; })
constexpr bool vall(vec<N, T1> const &t1) {
    bool ret = (bool)t1[0];
    for (int i = 1; i < N; i++) {
        ret = ret && (bool)t1[i];
    }
    return ret;
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 <=> t2; })
constexpr decltype(auto) vcmp(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 <=> t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 == t2; })
constexpr decltype(auto) vcmpeq(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 == t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 != t2; })
constexpr decltype(auto) vcmpne(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 != t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 >= t2; })
constexpr decltype(auto) vcmpge(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 >= t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 > t2; })
constexpr decltype(auto) vcmpgt(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 > t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 <= t2; })
constexpr decltype(auto) vcmple(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 <= t2; });
}


template <class T1, class T2>
    requires ((is_vec<T1> || is_vec<T2>) && requires (remove_vec_t<T1> t1, remove_vec_t<T2> t2) { t1 < t2; })
constexpr decltype(auto) vcmplt(T1 const &t1, T2 const &t2) {
    return vec_wise(t1, t2, [] (auto &&t1, auto &&t2) { return t1 < t2; });
}


}
ZENO_NAMESPACE_END
