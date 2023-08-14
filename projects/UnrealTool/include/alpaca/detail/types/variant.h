#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VARIANT
#include <alpaca/detail/type_info.h>
#include <alpaca/detail/variable_length_encoding.h>
#include <alpaca/detail/variant_nth_field.h>
#include <system_error>
#include <variant>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T, std::size_t N, std::size_t I>
void type_info_variant_helper(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  if constexpr (I < N) {

    // save current type
    type_info<std::variant_alternative_t<I, T>>(typeids, struct_visitor_map);

    // go to next type
    type_info_variant_helper<T, N, I + 1>(typeids, struct_visitor_map);
  }
}

template <typename T>
typename std::enable_if<is_specialization<T, std::variant>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::variant>());
  constexpr auto variant_size = std::variant_size_v<T>;
  type_info_variant_helper<T, variant_size, 0>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename Container, typename... U>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::variant<U...> &input) {
  std::size_t index = input.index();

  // save index of variant
  to_bytes_router<O, std::size_t>(index, bytes, byte_index);

  // save value of variant
  const auto visitor = [&bytes, &byte_index](auto &&arg) {
    to_bytes_router<O>(arg, bytes, byte_index);
  };
  std::visit(visitor, input);
}

template <options O, typename Container, typename... T>
bool from_bytes(std::variant<T...> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  // current byte is the index of the variant value
  std::size_t index = 0;
  detail::from_bytes<O, std::size_t>(index, bytes, byte_index, end_index,
                                     error_code);

  // read bytes as value_type = variant@index
  detail::set_variant_value<O, std::variant<T...>, Container>(
      output, index, bytes, byte_index, end_index, error_code);

  return true;
}

} // namespace detail

} // namespace alpaca
#endif