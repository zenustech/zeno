#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
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
    return tovec<glm::vec<N, T>>(a);
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
struct is_promotable<glm::vec<N, T>, glm::vec<N, T>> {
    static constexpr bool value = true;
    using type = glm::vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<glm::vec<N, T>, T> {
    static constexpr bool value = true;
    using type = glm::vec<N, T>;
};

template <class T, size_t N>
struct is_promotable<T, glm::vec<N, T>> {
    static constexpr bool value = true;
    using type = glm::vec<N, T>;
};

template <class T>
struct is_promotable<T, T> {
    static constexpr bool value = true;
    using type = T;
};

template <class T, class S>
constexpr bool is_promotable_v = is_promotable<T, S>::value;

template <class T, class S>
using is_promotable_t = typename is_promotable<T, S>::type;


template <class T, class S>
struct is_castable {
    static constexpr bool value = false;
};
template <class T, size_t N>
struct is_castable<glm::vec<N, T>, T> {
    static constexpr bool value = true;
};

template <class T, size_t N>
struct is_castable<T, glm::vec<N, T>> {
    static constexpr bool value = false;
};

template <class T>
struct is_castable<T, T> {
    static constexpr bool value = true;
};

template <class T, class S>
inline constexpr bool is_castable_v = is_castable<T, S>::value;

}
