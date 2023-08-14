#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_PAIR
#include <alpaca/detail/type_info.h>
#include <system_error>
#include <utility>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T>
typename std::enable_if<is_specialization<T, std::pair>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::pair>());

  using first_type = typename T::first_type;
  type_info<first_type>(typeids, struct_visitor_map);

  using second_type = typename T::second_type;
  type_info<second_type>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename Container, typename U, typename V>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::pair<U, V> &input) {
  to_bytes_router<O>(input.first, bytes, byte_index);
  to_bytes_router<O>(input.second, bytes, byte_index);
}

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename U, typename Container>
bool from_bytes(std::pair<T, U> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_router<O>(output.first, bytes, byte_index, end_index, error_code);
  from_bytes_router<O>(output.second, bytes, byte_index, end_index, error_code);

  return true;
}

} // namespace detail

} // namespace alpaca
#endif