#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNIQUE_PTR
#include <alpaca/detail/to_bytes.h>
#include <alpaca/detail/type_info.h>
#include <memory>
#include <system_error>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T>
typename std::enable_if<is_specialization<T, std::unique_ptr>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::unique_ptr>());
  using element_type = typename T::element_type;
  type_info<element_type>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename Container, typename U>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::unique_ptr<U> &input) {
  auto has_value = false;
  if (input) {
    has_value = true;
  }

  // save if ptr has value
  to_bytes_router<O, bool>(has_value, bytes, byte_index);

  // save value
  if (has_value) {
    to_bytes_router<O, U>(*input, bytes, byte_index);
  }
}

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename T, typename Container>
bool from_bytes(std::unique_ptr<T> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  auto current_byte = bytes[byte_index];

  // check if has_value has a legal value of either 0 or 1
  if (current_byte != 0x00 && current_byte != 0x01) {
    // expected either 0 or 1, got something else
    error_code = std::make_error_code(std::errc::illegal_byte_sequence);

    // stop here
    return false;
  }

  // current byte is the `has_value` byte
  bool has_value = static_cast<bool>(bytes[byte_index++]);

  if (has_value) {
    // read value of unique_ptr
    T value{};
    from_bytes_router<O>(value, bytes, byte_index, end_index, error_code);
    output = std::unique_ptr<T>(new T{std::move(value)});
  } else {
    output = nullptr;
  }

  return true;
}

} // namespace detail

} // namespace alpaca
#endif