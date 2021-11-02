#pragma once


#include "vec_type.h"
#include "vec_traits.h"
#include <tuple>


ZENO_NAMESPACE_BEGIN
namespace math {


template <is_not_vec T0, size_t N, is_not_vec T1>
    requires (requires (T1 t1) { (T0)t1; })
constexpr vec<N, T0> vcast(vec<N, T1> const &t1) {
    return (vec<N, T0>)t1;
}

template <is_not_vec T0, is_not_vec T1>
    requires (requires (T1 t1) { (T0)t1; })
constexpr T0 vcast(T1 const &t1) {
    return (T0)t1;
}


template <size_t N, class T>
constexpr auto totuple(vec<N, T> const &t) {
    return ([] <size_t ...Is> (vec<N, T> const &t, std::index_sequence<Is...>) {
        return std::make_tuple(t.template get<Is>()...);
    })(t, std::make_index_sequence<N>());
}

template <size_t N, class Tuple>
constexpr auto fromtuple(Tuple const &t) {
    using T = std::remove_cvref_t<decltype(std::get<0>(t))>;
    return ([] <size_t ...Is> (Tuple const &t, std::index_sequence<Is...>) {
        return vec<N, T>(std::get<Is>(t)...);
    })(t, std::make_index_sequence<N>());
}

template <class Other, size_t N, class T>
constexpr Other toother(vec<N, T> const &t) {
    return ([] <size_t ...Is> (vec<N, T> const &t, std::index_sequence<Is...>) {
        return Other(t.template get<Is>()...);
    })(t, std::make_index_sequence<N>());
}

template <size_t N, class Other>
constexpr auto fromother(Other const &t) {
    using T = std::remove_cvref_t<decltype(t[0])>;
    return ([] <size_t ...Is> (Other const &t, std::index_sequence<Is...>) {
        return vec<N, T>(t[Is]...);
    })(t, std::make_index_sequence<N>());
}


}
ZENO_NAMESPACE_END
