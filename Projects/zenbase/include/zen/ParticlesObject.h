#pragma once


#include <zen/zen.h>
#include <glm/vec3.hpp>
#include <vector>

namespace zen {

struct ParticlesObject : zen::IObject {
  
  std::vector<glm::vec3> pos;
  std::vector<glm::vec3> vel;

  size_t size() const {
    return pos.size();
  }
};

}
