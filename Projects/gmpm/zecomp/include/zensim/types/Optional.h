#pragma once
#include <optional>

namespace zs {

  template <typename T> using optional = std::optional<T>;
  using nullopt_t = std::nullopt_t;
  static constexpr auto nullopt = std::nullopt;

}  // namespace zs
