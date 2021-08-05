#pragma once

#include "common.h"
#include <cstdint>


namespace zeno::v2::container {

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

template <class T, class = void>
struct any_traits {
    using underlying_type = T;
};

template <class T>
struct any_traits<T, std::void_t<decltype(
        std::declval<scalar_type_variant &>() = T
        )>> {
    using underlying_type = scalar_type_variant;
};

template <class T>
T any_cast(any &&a) {
    std::visit([] (auto t) {
        return t;
    }, a);
}

}
