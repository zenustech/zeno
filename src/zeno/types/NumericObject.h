#pragma once

#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
#include <zeno/utils/variantswitch.h>
#include <variant>

namespace zeno {


using NumericValue = std::variant<
  int, zeno::vec2i, zeno::vec3i, zeno::vec4i,
  float, zeno::vec2f, zeno::vec3f, zeno::vec4f>;

struct NumericObject : IObjectClone<NumericObject> {
  NumericValue value;

  NumericObject() = default;
  NumericObject(NumericValue const &value) : value(value) {}

  NumericValue &get() {
      return value;
  }

  NumericValue const &get() const {
      return value;
  }

  template <class T>
  T get() const {
    if (!is<T>())
        throw TypeError(typeid(T), typeid_of_variant<NumericValue>(value), "NumericObject::get<T>");
    return std::get<T>(value);
  }

  template <class T>
  bool is() const {
    return std::holds_alternative<T>(value);
  }

  template <class T>
  void set(T const &x) {
    value = x;
  }
};


}
