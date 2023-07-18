#pragma once
#include <alpaca/detail/field_type.h>
#include <alpaca/detail/is_specialization.h>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_ARRAY
#include <array>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
#include <map>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNIQUE_PTR
#include <memory>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_OPTIONAL
#include <optional>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
#include <set>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_STRING
#include <string>
#endif

#include <string_view>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_TUPLE
#include <tuple>
#endif

#include <unordered_map>

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
#include <unordered_set>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_PAIR
#include <utility>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VARIANT
#include <variant>
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VECTOR
#include <vector>
#endif

namespace alpaca {

namespace detail {

template <typename T>
typename std::enable_if<std::is_same_v<T, bool>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::bool_>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, char>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::char_>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, uint8_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::uint8>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, uint16_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::uint16>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, uint32_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::uint32>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, uint64_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::uint64>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, int8_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::int8>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, int16_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::int16>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, int32_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::int32>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, int64_t>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::int64>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, float>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::float32>());
}

template <typename T>
typename std::enable_if<std::is_same_v<T, double>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::float64>());
}

template <typename T>
typename std::enable_if<std::is_enum_v<T>, void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &) {
  typeids.push_back(to_byte<field_type::enum_class>());
}

template <typename T> struct is_array_type : std::false_type {};

template <class T, std::size_t N>
struct is_array_type<std::array<T, N>> : std::true_type {};

// Forward declares

// aggregate types
template <typename T,
          std::size_t N = detail::aggregate_arity<std::remove_cv_t<T>>::size()>
typename std::enable_if<std::is_aggregate_v<T> && !is_array_type<T>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_ARRAY
// array types
template <typename T>
typename std::enable_if<is_array_type<T>::value, void>::type type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_MAP
// map
template <typename T>
typename std::enable_if<is_specialization<T, std::map>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_MAP
template <typename T>
typename std::enable_if<is_specialization<T, std::unordered_map>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_OPTIONAL
// optional
template <typename T>
typename std::enable_if<is_specialization<T, std::optional>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_PAIR
// pair
template <typename T>
typename std::enable_if<is_specialization<T, std::pair>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_SET
// set
template <typename T>
typename std::enable_if<is_specialization<T, std::set>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNORDERED_SET
template <typename T>
typename std::enable_if<is_specialization<T, std::unordered_set>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_STRING
// string
template <typename T>
typename std::enable_if<is_specialization<T, std::basic_string>::value,
                        void>::type
type_info(std::vector<uint8_t> &typeids,
          std::unordered_map<std::string_view, std::size_t> &);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_TUPLE
// tuple
template <typename T, std::size_t N, std::size_t I>
void type_info_tuple_helper(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_UNIQUE_PTR
// unique_ptr
template <typename T>
typename std::enable_if<is_specialization<T, std::unique_ptr>::value,
                        void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VARIANT
// variant
template <typename T>
typename std::enable_if<is_specialization<T, std::variant>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

#ifndef ALPACA_EXCLUDE_SUPPORT_STD_VECTOR
// vector
template <typename T>
typename std::enable_if<is_specialization<T, std::vector>::value, void>::type
type_info(
    std::vector<uint8_t> &typeids,
    std::unordered_map<std::string_view, std::size_t> &struct_visitor_map);
#endif

} // namespace detail

} // namespace alpaca