#pragma once


#include "vec_type.h"
#include <type_traits>


ZENO_NAMESPACE_BEGIN
namespace math {


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


}
ZENO_NAMESPACE_END


namespace std {

template <size_t N, class T, size_t I>
struct tuple_element<I, ZENO_NAMESPACE::math::vec<N, T>> {
    using type = T;
};

template <size_t N, class T>
struct tuple_size<ZENO_NAMESPACE::math::vec<N, T>>
    : integral_constant<size_t, N> {};

}
