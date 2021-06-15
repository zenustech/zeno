#pragma once

#include <zen/zen.h>
#include <zen/vec.h>
#include <variant>

namespace zen {


using NumericValue = std::variant<
  int, float, zen::vec2f, zen::vec3f, zen::vec4f>;

struct NumericObject : IObject {
  NumericValue value;

  template <class T>
  T get() {
    return std::get<T>(value);
  }

  template <class T>
  T is() {
    return std::holds_alternative<T>(value);
  }

  template <class T>
  void set(T const &x) {
    value = x;
  }
};

}
