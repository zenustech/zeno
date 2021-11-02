#pragma once

#include "vec.h"
#include <zeno/ztd/any_ptr.h>
#include "type_traits.h"


namespace zeno {

using scalar_type_variant = std::variant
        < bool
        , uint8_t
        , uint16_t
        , uint32_t
        , uint64_t
        , int8_t
        , int16_t
        , int32_t
        , int64_t
        , float
        , double
        >;

template <size_t N>
using vector_type_variant = std::variant
        < vec<N, bool>
        , vec<N, uint8_t>
        , vec<N, uint16_t>
        , vec<N, uint32_t>
        , vec<N, uint64_t>
        , vec<N, int8_t>
        , vec<N, int16_t>
        , vec<N, int32_t>
        , vec<N, int64_t>
        , vec<N, float>
        , vec<N, double>
        >;

template <class T>
using any_underlying_type_t = ZENO_NAMESPACE::ztd::any_underlying_t<T>;

using Any = ZENO_NAMESPACE::ztd::any_ptr;

template <class T>
std::optional<T> exact_any_cast(Any a) {
    if constexpr (is_shared_ptr<T>::value) {
        if (auto p = a.pointer_cast<typename remove_shaded_ptr<T>::type>()) {
            return std::make_optional(p);
        } else {
            return std::nullopt;
        }
    } else {
        if (auto p = a.pointer_cast<T>()) {
            return std::make_optional(*p);
        } else {
            return std::nullopt;
        }
    }

}

template <class T>
std::optional<T> silent_any_cast(Any const &a) {
    if constexpr (is_shared_ptr<T>::value) {
        if (auto p = a.pointer_cast<typename remove_shaded_ptr<T>::type>()) {
            return std::make_optional(p);
        } else {
            return std::nullopt;
        }
    } else {
        return a.value_cast<T>();
    }
}

using zany = Any;

}
