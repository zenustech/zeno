#pragma once


#include <zen/zen.h>
#include <glm/matrix.hpp>

namespace zenbase {

struct MatrixObject : zen::IObject {
  glm::mat4 m;

  glm::mat4 to_4x4() {
    return m;
  }
};

}
