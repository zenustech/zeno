#pragma once

#include <type_traits>
#include <utility>
#include <cstdint>

namespace zeno {

template <class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
constexpr std::underlying_type_t<T> to_underlying(T t) noexcept {
    return static_cast<std::underlying_type_t<T>>(t);
}

template <class T, std::enable_if_t<!std::is_enum_v<T>, int> = 0>
constexpr T to_underlying(T t) noexcept {
    return t;
}

}
