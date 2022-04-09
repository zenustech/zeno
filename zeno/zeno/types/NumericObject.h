#pragma once

#include <zeno/utils/vec.h>
#include <zeno/core/IObject.h>
#include <zeno/utils/Exception.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
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
  T get() const {
    return std::visit([] (auto const &val) -> T {
        using V = std::decay_t<decltype(val)>;
        if constexpr (!std::is_constructible_v<T, V>) {
            throw Exception((std::string)"NumericObject expect `" + typeid(T).name() + "`, got `" + typeid(V).name());
        } else {
            return T(val);
        }
    }, value);
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

struct MatrixObject : zeno::IObjectClone<MatrixObject>{//ZhxxHappyObject
    std::variant<glm::mat3, glm::mat4> m;
};

}
