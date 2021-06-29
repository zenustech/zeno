#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>
#include <variant>

namespace zeno {


using NumericValue = std::variant<
  int, float, zeno::vec2f, zeno::vec3f, zeno::vec4f>;

struct NumericObject : IObjectClone<NumericObject> {
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
