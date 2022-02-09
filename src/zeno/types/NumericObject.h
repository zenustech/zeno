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
        throw Exception((std::string)"NumericObject expect `" + typeid(T).name()
                + "`, got index `" + std::to_string(value.index()) + "`");
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
