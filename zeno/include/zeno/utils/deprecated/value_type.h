#pragma once

#include <type_traits>

namespace zeno {

template <class T>
struct type_identity {
    using type = T;
};

template <class T>
static std::decay_t<T> decay_of(T &&t) {
    return {};
}

template <class T>
static type_identity<std::decay_t<T>> decay_identity(T &&t) {
    return {};
}

template <class T>
static typename std::decay_t<T>::value_type value_type_of(T &&t) {
    return {};
}

template <class T>
static type_identity<typename std::decay_t<T>::value_type> value_type_identity(T &&t) {
    return {};
}

}
