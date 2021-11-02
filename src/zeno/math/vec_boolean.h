#pragma once


#include "vec_elmwise.h"


ZENO_NAMESPACE_BEGIN
namespace math {


template <class T>
struct vbool {
    T t;

    constexpr vbool(T const &t) : t(t) {
    }

    constexpr operator T const &() const {
        return t;
    }

    constexpr operator T &() {
        return t;
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1) { !t1; })
    constexpr vbool operator!() const {
        return vec_wise(t, [] (auto &&t1) { return !t1; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 && t2; })
    constexpr vbool operator&&(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 && t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 || t2; })
    constexpr vbool operator||(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 || t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 <=> t2; })
    constexpr vbool operator<=>(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 <=> t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 == t2; })
    constexpr vbool operator==(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 == t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 != t2; })
    constexpr vbool operator!=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 != t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 >= t2; })
    constexpr vbool operator>=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 >= t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 > t2; })
    constexpr vbool operator>(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 > t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 <= t2; })
    constexpr vbool operator<=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 <= t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 < t2; })
    constexpr vbool operator<(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 < t2; });
    }
};

template <class T>
vbool(T const &) -> vbool<T>;


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


}
ZENO_NAMESPACE_END
