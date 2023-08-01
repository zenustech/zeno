#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VARIANT
#include <alpaca/detail/options.h>
#include <cstdint>
#include <variant>
#include <vector>

namespace alpaca {

namespace detail {

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename type, typename Container,
          std::size_t variant_size = std::variant_size_v<type>>
constexpr void set_variant_value(type &variant, std::size_t index,
                                 Container &bytes, std::size_t &byte_index,
                                 std::size_t &end_index,
                                 std::error_code &error_code) noexcept {
  if constexpr (variant_size == 1) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 2) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 3) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 4) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 5) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 6) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 7) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 8) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 9) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 10) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 11) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 12) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 13) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 14) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 15) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 16) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 17) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 18) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 19) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 20) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 21) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 22) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 23) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 24) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 25) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 26) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 27) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 28) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 29) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 30) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 31) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 32) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 33) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 34) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 35) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 36) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 37) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 38) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 39) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 40) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 41) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 42) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 43) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 44) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 45) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 46) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 47) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 48) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 49) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 50) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 51) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 52) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 53) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 54) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 55) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 56) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 57) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 58) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 59) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 60) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 61) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 62) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 63) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 64) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 65) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 66) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 67) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 68) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 69) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 70) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 71) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 72) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 73) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 74) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 75) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 76) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 77) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 78) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 79) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 80) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 81) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 82) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 83) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 84) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 85) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 86) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 87) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 88) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 89) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 90) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 91) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 92) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 93) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 94) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 95) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 94) {
      typename std::variant_alternative_t<94, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 96) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 94) {
      typename std::variant_alternative_t<94, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 95) {
      typename std::variant_alternative_t<95, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 97) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 94) {
      typename std::variant_alternative_t<94, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 95) {
      typename std::variant_alternative_t<95, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 96) {
      typename std::variant_alternative_t<96, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 98) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 94) {
      typename std::variant_alternative_t<94, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 95) {
      typename std::variant_alternative_t<95, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 96) {
      typename std::variant_alternative_t<96, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 97) {
      typename std::variant_alternative_t<97, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else if constexpr (variant_size == 99) {
    if (index == 0) {
      typename std::variant_alternative_t<0, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 1) {
      typename std::variant_alternative_t<1, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 2) {
      typename std::variant_alternative_t<2, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 3) {
      typename std::variant_alternative_t<3, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 4) {
      typename std::variant_alternative_t<4, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 5) {
      typename std::variant_alternative_t<5, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 6) {
      typename std::variant_alternative_t<6, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 7) {
      typename std::variant_alternative_t<7, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 8) {
      typename std::variant_alternative_t<8, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 9) {
      typename std::variant_alternative_t<9, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 10) {
      typename std::variant_alternative_t<10, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 11) {
      typename std::variant_alternative_t<11, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 12) {
      typename std::variant_alternative_t<12, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 13) {
      typename std::variant_alternative_t<13, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 14) {
      typename std::variant_alternative_t<14, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 15) {
      typename std::variant_alternative_t<15, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 16) {
      typename std::variant_alternative_t<16, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 17) {
      typename std::variant_alternative_t<17, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 18) {
      typename std::variant_alternative_t<18, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 19) {
      typename std::variant_alternative_t<19, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 20) {
      typename std::variant_alternative_t<20, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 21) {
      typename std::variant_alternative_t<21, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 22) {
      typename std::variant_alternative_t<22, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 23) {
      typename std::variant_alternative_t<23, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 24) {
      typename std::variant_alternative_t<24, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 25) {
      typename std::variant_alternative_t<25, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 26) {
      typename std::variant_alternative_t<26, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 27) {
      typename std::variant_alternative_t<27, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 28) {
      typename std::variant_alternative_t<28, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 29) {
      typename std::variant_alternative_t<29, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 30) {
      typename std::variant_alternative_t<30, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 31) {
      typename std::variant_alternative_t<31, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 32) {
      typename std::variant_alternative_t<32, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 33) {
      typename std::variant_alternative_t<33, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 34) {
      typename std::variant_alternative_t<34, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 35) {
      typename std::variant_alternative_t<35, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 36) {
      typename std::variant_alternative_t<36, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 37) {
      typename std::variant_alternative_t<37, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 38) {
      typename std::variant_alternative_t<38, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 39) {
      typename std::variant_alternative_t<39, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 40) {
      typename std::variant_alternative_t<40, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 41) {
      typename std::variant_alternative_t<41, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 42) {
      typename std::variant_alternative_t<42, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 43) {
      typename std::variant_alternative_t<43, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 44) {
      typename std::variant_alternative_t<44, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 45) {
      typename std::variant_alternative_t<45, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 46) {
      typename std::variant_alternative_t<46, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 47) {
      typename std::variant_alternative_t<47, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 48) {
      typename std::variant_alternative_t<48, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 49) {
      typename std::variant_alternative_t<49, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 50) {
      typename std::variant_alternative_t<50, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 51) {
      typename std::variant_alternative_t<51, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 52) {
      typename std::variant_alternative_t<52, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 53) {
      typename std::variant_alternative_t<53, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 54) {
      typename std::variant_alternative_t<54, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 55) {
      typename std::variant_alternative_t<55, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 56) {
      typename std::variant_alternative_t<56, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 57) {
      typename std::variant_alternative_t<57, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 58) {
      typename std::variant_alternative_t<58, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 59) {
      typename std::variant_alternative_t<59, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 60) {
      typename std::variant_alternative_t<60, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 61) {
      typename std::variant_alternative_t<61, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 62) {
      typename std::variant_alternative_t<62, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 63) {
      typename std::variant_alternative_t<63, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 64) {
      typename std::variant_alternative_t<64, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 65) {
      typename std::variant_alternative_t<65, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 66) {
      typename std::variant_alternative_t<66, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 67) {
      typename std::variant_alternative_t<67, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 68) {
      typename std::variant_alternative_t<68, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 69) {
      typename std::variant_alternative_t<69, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 70) {
      typename std::variant_alternative_t<70, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 71) {
      typename std::variant_alternative_t<71, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 72) {
      typename std::variant_alternative_t<72, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 73) {
      typename std::variant_alternative_t<73, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 74) {
      typename std::variant_alternative_t<74, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 75) {
      typename std::variant_alternative_t<75, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 76) {
      typename std::variant_alternative_t<76, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 77) {
      typename std::variant_alternative_t<77, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 78) {
      typename std::variant_alternative_t<78, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 79) {
      typename std::variant_alternative_t<79, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 80) {
      typename std::variant_alternative_t<80, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 81) {
      typename std::variant_alternative_t<81, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 82) {
      typename std::variant_alternative_t<82, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 83) {
      typename std::variant_alternative_t<83, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 84) {
      typename std::variant_alternative_t<84, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 85) {
      typename std::variant_alternative_t<85, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 86) {
      typename std::variant_alternative_t<86, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 87) {
      typename std::variant_alternative_t<87, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 88) {
      typename std::variant_alternative_t<88, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 89) {
      typename std::variant_alternative_t<89, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 90) {
      typename std::variant_alternative_t<90, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 91) {
      typename std::variant_alternative_t<91, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 92) {
      typename std::variant_alternative_t<92, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 93) {
      typename std::variant_alternative_t<93, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 94) {
      typename std::variant_alternative_t<94, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 95) {
      typename std::variant_alternative_t<95, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 96) {
      typename std::variant_alternative_t<96, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 97) {
      typename std::variant_alternative_t<97, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else if (index == 98) {
      typename std::variant_alternative_t<98, type> value{};
      from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
      variant = value;
    } else {
      return;
    }
  } else /* extend it by yourself for higher arities */ {
    return;
  }
}

} // namespace detail

} // namespace alpaca
#endif