#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_TUPLE
#include <alpaca/detail/type_info.h>
#include <system_error>
#include <tuple>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T, std::size_t N, std::size_t I>
void type_info_tuple_helper(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  if constexpr (I < N) {

    // save current type
    type_info<typename std::tuple_element<I, T>::type>(typeids,
                                                       struct_visitor_map);

    // go to next type
    type_info_tuple_helper<T, N, I + 1>(typeids, struct_visitor_map);
  }
}

template <typename T>
typename std::enable_if<is_specialization<T, std::tuple>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::tuple>());
  constexpr auto tuple_size = std::tuple_size_v<T>;
  type_info_tuple_helper<T, tuple_size, 0>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename T, typename Container, std::size_t index>
void save_tuple_value(const T &tuple, Container &bytes,
                      std::size_t &byte_index) {
  constexpr auto max_index = std::tuple_size<T>::value;
  if constexpr (index < max_index) {
    to_bytes_router<O>(std::get<index>(tuple), bytes, byte_index);
    save_tuple_value<O, T, Container, index + 1>(tuple, bytes, byte_index);
  }
}

template <options O, typename T, typename Container>
void to_bytes_from_tuple_type(const T &input, Container &bytes,
                              std::size_t &byte_index) {
  // value of each element
  save_tuple_value<O, T, Container, 0>(input, bytes, byte_index);
}

template <options O, typename Container, typename... U>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::tuple<U...> &input) {
  to_bytes_from_tuple_type<O>(input, bytes, byte_index);
}

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename Container, std::size_t index>
void load_tuple_value(T &tuple, Container &bytes, std::size_t &current_index,
                      std::size_t &end_index, std::error_code &error_code) {
  constexpr auto max_index = std::tuple_size<T>::value;
  if constexpr (index < max_index) {
    from_bytes_router<O>(std::get<index>(tuple), bytes, current_index,
                         end_index, error_code);
    load_tuple_value<O, T, Container, index + 1>(tuple, bytes, current_index,
                                                 end_index, error_code);
  }
}

template <options O, typename T, typename Container>
void from_bytes_to_tuple(T &tuple, Container &bytes, std::size_t &current_index,
                         std::size_t &end_index, std::error_code &error_code) {
  load_tuple_value<O, T, Container, 0>(tuple, bytes, current_index, end_index,
                                       error_code);
}

template <options O, typename Container, typename... T>
bool from_bytes(std::tuple<T...> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_tuple<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}

} // namespace detail

} // namespace alpaca
#endif