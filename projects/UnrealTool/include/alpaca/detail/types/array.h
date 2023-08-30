#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_ARRAY
#include <alpaca/detail/type_info.h>
#include <array>
#include <system_error>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T>
typename std::enable_if<is_array_type<T>::value, void>::type type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::array>());
  typeids.push_back(std::tuple_size_v<T>);
  using value_type = typename T::value_type;
  type_info<value_type>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename Container, typename T, std::size_t N>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::array<T, N> &input) {
  // value of each element in list
  for (const auto &v : input) {
    to_bytes_router<O>(v, bytes, byte_index);
  }
}

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename Container>
void from_bytes_to_array(T &value, Container &bytes, std::size_t &current_index,
                         std::size_t &end_index, std::error_code &error_code) {

  using decayed_value_type = typename std::decay<typename T::value_type>::type;

  constexpr auto size = std::tuple_size<T>::value;

  if (size > end_index - current_index) {
    // size is greater than the number of bytes remaining
    error_code = std::make_error_code(std::errc::value_too_large);

    // stop here
    return;
  }

  // read `size` bytes and save to value
  for (std::size_t i = 0; i < size; ++i) {
    decayed_value_type v{};
    from_bytes_router<O>(v, bytes, current_index, end_index, error_code);
    value[i] = v;
  }
}

template <options O, typename U, typename Container, std::size_t N>
bool from_bytes(std::array<U, N> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_array<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}

} // namespace detail

} // namespace alpaca
#endif