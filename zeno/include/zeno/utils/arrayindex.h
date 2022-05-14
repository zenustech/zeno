#pragma once

#include <array>
#include <algorithm>
#include <initializer_list>
#include <zeno/utils/Error.h>
#include <zeno/utils/to_string.h>

namespace zeno {

template <class T, std::size_t N>
std::array<T, N> to_array(T const a[N]) {
    return std::array<T, N>{a};
}

template <class T0, class ...Ts>
std::array<T0, sizeof...(Ts) + 1> make_array(T0 const &t0, Ts const &...ts) {
    return {t0, T0(ts)...};
}

template <class T, class V>
std::size_t array_index(T const &arr, V const &val) {
    auto it = std::find(std::begin(arr), std::end(arr), val);
    return it - std::begin(arr);
}

template <class T, class V>
std::size_t array_index_safe(T const &arr, V const &val, std::string const &type) {
    auto it = std::find(std::begin(arr), std::end(arr), val);
    if (it == std::end(arr))
        throw makeError<KeyError>(to_string(val), type);
    return it - std::begin(arr);
}

template <class T, class V>
std::size_t array_index(std::initializer_list<T> arr, V const &val) {
    auto it = std::find(std::begin(arr), std::end(arr), val);
    return it - std::begin(arr);
}

template <class T, class V>
std::size_t array_index_safe(std::initializer_list<T> arr, V const &val, std::string const &type) {
    auto it = std::find(std::begin(arr), std::end(arr), val);
    if (it == std::end(arr))
        throw makeError<KeyError>(to_string(val), type);
    return it - std::begin(arr);
}

}
