#pragma once
#ifndef ALPACA_EXCLUDE_SUPPORT_STD_CHRONO
#include <alpaca/detail/options.h>
#include <alpaca/detail/type_info.h>
#include <chrono>
#include <system_error>
#include <vector>

namespace alpaca {

namespace detail {

template <typename T>
typename std::enable_if<is_specialization<T, std::chrono::duration>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map) {
  typeids.push_back(to_byte<field_type::chrono_duration>());

  // save the rep type of duration
  using rep = typename T::rep;
  type_info<rep>(typeids, struct_visitor_map);
}

template <options O, typename T, typename Container>
void to_bytes_router(const T &input, Container &bytes, std::size_t &byte_index);

template <options O, typename Container, typename Rep, typename Period>
void to_bytes(Container &bytes, std::size_t &byte_index,
              const std::chrono::duration<Rep, Period> &input) {
  // save the count
  to_bytes_router<O>(input.count(), bytes, byte_index);
}

template <options O, typename T, typename Container>
void from_bytes_router(T &output, Container &bytes, std::size_t &byte_index,
                       std::size_t &end_index, std::error_code &error_code);

template <options O, typename Rep, typename Period, typename Container>
bool from_bytes(std::chrono::duration<Rep, Period> &output, Container &bytes,
                std::size_t &byte_index, std::size_t &end_index,
                std::error_code &error_code) {

  if (byte_index >= end_index) {
    // end of input
    // return true for forward compatibility
    return true;
  }

  Rep count{0};
  from_bytes_router<O>(count, bytes, byte_index, end_index, error_code);

  // Use explicit constructor
  //   constexpr explicit duration( const Rep2& r );
  // to construct the duration type
  output = std::chrono::duration<Rep, Period>{count};

  return true;
}

} // namespace detail

} // namespace alpaca
#endif