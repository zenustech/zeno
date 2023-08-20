#pragma once
#include <alpaca/detail/to_bytes.h>
#include <alpaca/detail/type_info.h>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
#include <set>
#endif

#include <system_error>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
#include <unordered_set>
#endif

#include <vector>

namespace alpaca {

namespace detail {

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
template <typename T>
typename std::enable_if<is_specialization<T, std::set>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::set>());
  using value_type = typename T::value_type;
  type_info<value_type>(typeids, struct_visitor_map);
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
template <typename T>
typename std::enable_if<is_specialization<T, std::unordered_set>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::unordered_set>());
  using value_type = typename T::value_type;
  type_info<value_type>(typeids, struct_visitor_map);
}
#endif

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename T, typename Container>
void to_bytes_from_set_type(const T &input, Container &bytes,
                            std::size_t &byte_index) {
  // save set size
  to_bytes_router<O, std::size_t, Container>(input.size(), bytes, byte_index);

  // save values in set
  for (const auto &value : input) {
    to_bytes_router<O>(value, bytes, byte_index);
  }
}

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
template <options O, typename Container, typename U>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::set<U> &input) {
  to_bytes_from_set_type<O>(input, bytes, byte_index);
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
template <options O, typename Container, typename U>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::unordered_set<U> &input) {
  to_bytes_from_set_type<O>(input, bytes, byte_index);
}
#endif

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename Container>
void from_bytes_to_set(T &set, Container &bytes, std::size_t &current_index,
                       std::size_t &end_index, std::error_code &error_code) {
  // current byte is the size of the set
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
    typename T::value_type value{};
    from_bytes_router<O>(value, bytes, current_index, end_index, error_code);
    set.insert(value);
  }
}

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
template <options O, typename T, typename Container>
bool from_bytes(std::set<T> &output, Container &bytes, std::size_t &byte_index,
                std::size_t &end_index, std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_set<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
template <options O, typename T, typename Container>
bool from_bytes(std::unordered_set<T> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  from_bytes_to_set<O>(output, bytes, byte_index, end_index, error_code);
  return true;
}
#endif

} // namespace detail

} // namespace alpaca