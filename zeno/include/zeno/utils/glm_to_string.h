#pragma once

#include <zeno/utils/to_string.h>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <ostream>

namespace glm {

template <length_t L, class T, qualifier Q>
std::ostream &operator<<(std::ostream &os, vec<L, T, Q> const &v) {
    os << '{';
    os << v[0];
    for (std::size_t i = 1; i < L; i++) {
        os << ',' << v[i];
    }
    os << '}';
    return os;
}

template <length_t C, length_t R, class T, qualifier Q>
std::ostream &operator<<(std::ostream &os, mat<C, R, T, Q> const &v) {
    os << '{' << '{';
    os << v[0][0];
    for (std::size_t j = 1; j < R; j++) {
        os << ',' << v[0][j];
    }
    os << '}';
    for (std::size_t i = 1; i < C; i++) {
        os << ',' << '{';
        os << v[i][0];
        for (std::size_t j = 1; j < R; j++) {
            os << ',' << v[i][j];
        }
        os << '}';
    }
    os << '}';
    return os;
}

}

#if 0
namespace glm {

template <size_t I, ::glm::length_t L, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < L)>>
T const &get(::glm::vec<L, T, Q> const &v) {
    return v[I];
}

template <size_t I, ::glm::length_t L, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < L)>>
T &get(::glm::vec<L, T, Q> &v) {
    return v[I];
}

template <size_t I, ::glm::length_t L, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < L)>>
T &&get(::glm::vec<L, T, Q> &&v) {
    return std::move(v[I]);
}

template <size_t I, ::glm::length_t C, ::glm::length_t R, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < C * R)>>
T const &get(::glm::mat<C, R, T, Q> const &v) {
    return v[I % C][I / C];
}

template <size_t I, ::glm::length_t C, ::glm::length_t R, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < C * R)>>
T &get(::glm::mat<C, R, T, Q> &v) {
    return v[I % C][I / C];
}

template <size_t I, ::glm::length_t C, ::glm::length_t R, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < C * R)>>
T &&get(::glm::mat<C, R, T, Q> &&v) {
    return std::move(v[I % C][I / C]);
}

}

namespace std {

template <::glm::length_t L, class T, ::glm::qualifier Q>
struct tuple_size<::glm::vec<L, T, Q>> : integral_constant<size_t, static_cast<size_t>(L)> {
};

template <size_t I, ::glm::length_t L, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < L)>>
struct tuple_element<I, ::glm::vec<L, T, Q>> {
    using type = T;
};

template <::glm::length_t C, ::glm::length_t R, class T, ::glm::qualifier Q>
struct tuple_size<::glm::mat<C, R, T, Q>> : integral_constant<size_t, static_cast<size_t>(C * R)> {
};

template <size_t I, ::glm::length_t C, ::glm::length_t R, class T, ::glm::qualifier Q> //, class = std::enable_if_t<(I < C * R)>>
struct tuple_element<I, ::glm::mat<C, R, T, Q>> {
    using type = T;
};

using glm::get;

}
#endif
