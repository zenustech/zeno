#pragma once
#include <alpaca/detail/type_info.h>
#include <alpaca/detail/variable_length_encoding.h>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
#include <map>
#endif

#include <system_error>

#include <unordered_map>
#include <vector>

namespace alpaca {

namespace detail {

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
template <typename T>
typename std::enable_if<is_specialization<T, std::map>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::map>());
  using key_type = typename T::key_type;
  type_info<key_type>(typeids, struct_visitor_map);
  using mapped_type = typename T::mapped_type;
  type_info<mapped_type>(typeids, struct_visitor_map);
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_MAP
template <typename T>
typename std::enable_if<is_specialization<T, std::unordered_map>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::unordered_map>());
  using key_type = typename T::key_type;
  type_info<key_type>(typeids, struct_visitor_map);
  using mapped_type = typename T::mapped_type;
  type_info<mapped_type>(typeids, struct_visitor_map);
}
#endif

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename T, typename Container>
void to_bytes_from_map_type(const T &input, Container &bytes,
                            std::size_t &byte_index) {
  // save map size
  to_bytes_router<O, std::size_t, Container>(input.size(), bytes, byte_index);

  // save key,value pairs in map
  for (const auto &[key, value] : input) {
    to_bytes_router<O>(key, bytes, byte_index);
    to_bytes_router<O>(value, bytes, byte_index);
  }
}

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
template <options O, typename Container, typename K, typename V>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::map<K, V> &input) {
  to_bytes_from_map_type<O>(input, bytes, byte_index);
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_MAP
template <options O, typename Container, typename K, typename V>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::unordered_map<K, V> &input) {
  to_bytes_from_map_type<O>(input, bytes, byte_index);
}
#endif

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename Container>
void from_bytes_to_map(T &map, Container &bytes, std::size_t &current_index,
                       std::size_t &end_index, std::error_code &error_code) {
  // current byte is the size of the map
  std::size_t size = 0;
  detail::from_bytes<O, std::size_t>(size, bytes, current_index, end_index,
                                     error_code);

  if (size > end_index - current_index) {
    // size is greater than the number of bytes remaining
    error_code = std::make_error_code(std::errc::value_too_large);

    // stop here
    return;
  }

  // read `size` bytes and save to value
  for (std::size_t i = 0; i < size; ++i) {
    typename T::key_type key{};
    from_bytes_router<O>(key, bytes, current_index, end_index, error_code);

    typename T::mapped_type value{};
    from_bytes_router<O>(value, bytes, current_index, end_index, error_code);

    map.insert(std::make_pair(key, value));
  }
}

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
template <options O, typename K, typename V, typename Container>
bool from_bytes(std::map<K, V> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_map<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_MAP
template <options O, typename K, typename V, typename Container>
bool from_bytes(std::unordered_map<K, V> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_map<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}
#endif

} // namespace detail

} // namespace alpaca