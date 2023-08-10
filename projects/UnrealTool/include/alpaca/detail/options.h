#pragma once
#include <type_traits>

namespace alpaca {

enum class options {
  none = 0,
  big_endian = 1,
  fixed_length_encoding = 2,
  with_version = 4,
  with_checksum = 8
};

template <typename E> struct enable_bitmask_operators {
  static constexpr bool enable = false;
};

template <typename E>
constexpr typename std::enable_if<enable_bitmask_operators<E>::enable, E>::type
operator|(E lhs, E rhs) {
  using underlying = typename std::underlying_type<E>::type;
  return static_cast<E>(static_cast<underlying>(lhs) |
                        static_cast<underlying>(rhs));
}

namespace detail {

template <typename T, T value, T flag> constexpr bool enum_has_flag() {
  using underlying = typename std::underlying_type<T>::type;
  return (static_cast<underlying>(value) & static_cast<underlying>(flag)) ==
         static_cast<underlying>(flag);
}

template <options O> constexpr bool big_endian() {
  return enum_has_flag<options, O, options::big_endian>();
}

template <options O> constexpr bool little_endian() { return !big_endian<O>(); }

template <options O> constexpr bool fixed_length_encoding() {
  return enum_has_flag<options, O, options::fixed_length_encoding>();
}

template <options O> constexpr bool with_version() {
  return enum_has_flag<options, O, options::with_version>();
}

template <options O> constexpr bool with_checksum() {
  return enum_has_flag<options, O, options::with_checksum>();
}

} // namespace detail

template <> struct enable_bitmask_operators<options> {
  static constexpr bool enable = true;
};

} // namespace alpaca