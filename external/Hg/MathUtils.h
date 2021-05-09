#pragma once

#include "Vec.h"
#include <array>


namespace hg {

template <class VecT, class T>
VecT tovec(std::array<T, 1> const &a) {
    return {a[0]};
}

template <class VecT, class T>
VecT tovec(std::array<T, 2> const &a) {
    return {a[0], a[1]};
}

template <class VecT, class T>
VecT tovec(std::array<T, 3> const &a) {
    return {a[0], a[1], a[2]};
}

template <class VecT, class T>
VecT tovec(std::array<T, 4> const &a) {
    return {a[0], a[1], a[2], a[3]};
}

template <class T, size_t N>
auto tovec(std::array<T, N> const &a) {
    return tovec<Vec<N, T>>(a);
}

template <class T>
auto tovec(T const &a) {
    return a;
}


template <class T, class S>
struct is_promotable {
    static constexpr bool value = false;
    using type = void;
};

template <class T, size_t N>
struct is_promotable<Vec<N, T>, Vec<N, T>> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<Vec<N, T>, T> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<T, Vec<N, T>> {
    static constexpr bool value = true;
    using type = Vec<N, T>;
};

template <class T>
struct is_promotable<T, T> {
    static constexpr bool value = true;
    using type = T;
};

template <class T, class S>
inline constexpr bool is_promotable_v = is_promotable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T, class S>
using is_promotable_t = typename is_promotable<std::decay_t<T>, std::decay_t<S>>::type;


template <class T, class S>
struct is_castable {
    static constexpr bool value = false;
};
template <class T, size_t N>
struct is_castable<Vec<N, T>, T> {
    static constexpr bool value = true;
};

template <class T, size_t N>
struct is_castable<T, Vec<N, T>> {
    static constexpr bool value = false;
};

template <class T>
struct is_castable<T, T> {
    static constexpr bool value = true;
};

template <class T, class S>
inline constexpr bool is_castable_v = is_castable<std::decay_t<T>, std::decay_t<S>>::value;

template <class T, class S>
inline constexpr bool is_decay_same_v = std::is_same_v<std::decay_t<T>, std::decay_t<S>>;


template <class T>
auto typenameof() {
    return typeid(T).name();
};

template <class T>
auto typenameof(T const &_) {
    return typeid(T).name();
};

}
