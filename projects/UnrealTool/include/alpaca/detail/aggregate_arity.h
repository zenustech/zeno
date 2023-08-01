#pragma once
#include <utility>

namespace alpaca {

namespace detail {

struct filler {
  template <typename type> operator type();
};

template <typename aggregate, typename index_sequence = std::index_sequence<>,
          typename = void>
struct aggregate_arity : index_sequence {};

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

template <typename aggregate, std::size_t... indices>
struct aggregate_arity<
    aggregate, std::index_sequence<indices...>,
    std::void_t<decltype(aggregate{(void(indices), std::declval<filler>())...,
                                   std::declval<filler>()})>>
    : aggregate_arity<aggregate,
                      std::index_sequence<indices..., sizeof...(indices)>> {};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

} // namespace detail

} // namespace alpaca