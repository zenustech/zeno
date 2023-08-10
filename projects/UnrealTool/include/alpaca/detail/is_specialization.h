#pragma once
#include <type_traits>

namespace alpaca {

namespace detail {

// check if T is instantiation of U
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

} // namespace detail

} // namespace alpaca