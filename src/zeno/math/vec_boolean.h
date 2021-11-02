#pragma once


#include "vec_elmwise.h"


ZENO_NAMESPACE_BEGIN
namespace math {


template <class T>
struct vbool_t {
    T t;

    constexpr vbool_t(T const &t) : t(t) {
    }

    constexpr operator T const &() const {
        return t;
    }

    constexpr operator T &() {
        return t;
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1) { !t1; })
    constexpr auto operator!() const {
        return vec_wise(t, [] (auto &&t1) { return !t1; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 && t2; })
    constexpr auto operator&&(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 && t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 || t2; })
    constexpr auto operator||(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 || t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 <=> t2; })
    constexpr auto operator<=>(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 <=> t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 == t2; })
    constexpr auto operator==(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 == t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 != t2; })
    constexpr auto operator!=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 != t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 >= t2; })
    constexpr auto operator>=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 >= t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 > t2; })
    constexpr auto operator>(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 > t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 <= t2; })
    constexpr auto operator<=(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 <= t2; });
    }

    template <class T2>
        requires (vec_promotable<T, T2> && requires (remove_vec_t<T> t1, remove_vec_t<T2> t2) { t1 < t2; })
    constexpr auto operator<(T2 const &t2) const {
        return vec_wise(t, t2, [] (auto &&t1, auto &&t2) { return t1 < t2; });
    }

    template <size_t = 0>
        requires (is_vec<T> && requires (remove_vec_t<T> t1) { (bool)t1; })
    constexpr bool operator+() const {
        bool ret = (bool)t[0];
        for (int i = 1; i < vec_dimension_v<T>; i++) {
            ret = ret || (bool)t[i];
        }
        return ret;
    }

    template <size_t = 0>
        requires (is_vec<T> && requires (remove_vec_t<T> t1) { (bool)t1; })
    constexpr bool operator-() const {
        bool ret = (bool)t[0];
        for (int i = 1; i < vec_dimension_v<T>; i++) {
            ret = ret && (bool)t[i];
        }
        return ret;
    }

    template <size_t = 0>
        requires (is_not_vec<T> && requires (T t1) { (bool)t1; })
    constexpr bool operator+() const {
        return (bool)t;
    }

    template <size_t = 0>
        requires (is_not_vec<T> && requires (T t1) { (bool)t1; })
    constexpr bool operator-() const {
        return (bool)t;
    }
};

template <class T>
constexpr vbool_t<T> vbool(T const &t) {
    return vbool_t<T>(t);
}


}
ZENO_NAMESPACE_END
