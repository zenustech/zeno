#pragma once
#include "zensim/types/Value.h"

namespace zs {

  template <typename Object> struct BuilderFor {
  protected:
    Object &object;
    explicit constexpr BuilderFor(Object &obj) : object{obj} {}

  public:
    constexpr Object &target() noexcept { return object; }
    constexpr operator Object() noexcept { return std::move(object); }
  };

}  // namespace zs