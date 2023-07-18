#pragma once
#include <cstdint>

namespace alpaca {

namespace detail {

enum class field_type : uint8_t {
  bool_,
  char_,
  uint8,
  uint16,
  uint32,
  uint64,
  int8,
  int16,
  int32,
  int64,
  float32,
  float64,
  enum_class,
  string,
  array,
  vector,
  map,
  unordered_map,
  set,
  unordered_set,
  optional,
  pair,
  tuple,
  variant,
  unique_ptr,
  struct_,
  chrono_duration,
  list,
  deque
};

template <field_type value> constexpr uint8_t to_byte() {
  return static_cast<uint8_t>(value);
}

} // namespace detail

} // namespace alpaca