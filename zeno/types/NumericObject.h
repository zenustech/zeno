#pragma once

#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
#include <variant>

namespace zeno {


using NumericValue = std::variant<
  int, zeno::vec2i, zeno::vec3i, zeno::vec4i,
  float, zeno::vec2f, zeno::vec3f, zeno::vec4f>;

struct NumericObject : IObjectClone<NumericObject> {
  NumericValue value;

  NumericObject() = default;
  NumericObject(NumericValue value) : value(value) {}

  template <class T>
  T get() {
    if (!is<T>())
        throw Exception((std::string)"NumericObject expect `" + typeid(T).name()
                + "`, got index `" + "0123456789abcdefghijklmnopqrstuvwxyz"[value.index()] + "`");
    return std::get<T>(value);
  }

  template <class T>
  bool is() {
    return std::holds_alternative<T>(value);
  }

  template <class T>
  void set(T const &x) {
    value = x;
  }
};

}
