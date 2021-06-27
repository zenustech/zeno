#pragma once

#include "Meta.h"

namespace zs {

  struct wrapper_base {
    static constexpr bool is_same(void *) { return false; };
  };
  template <typename T> struct wrapper : wrapper_base {
    using wrapper_base::is_same;
    static constexpr bool is_same(wrapper<T> *) { return true; };
  };
  template <typename T1, typename T2> using is_same
      = std::integral_constant<bool, wrapper<T1>::is_same((wrapper<T2> *)nullptr)>;

  template <typename T1, typename T2> static constexpr auto is_same_v = is_same<T1, T2>::value;

}  // namespace zs
