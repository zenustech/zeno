#pragma once


#include <zeno/zeno.h>
#include <glm/vec3.hpp>
#include <vector>

namespace zeno {

struct ParticlesObject : zeno::IObjectClone<ParticlesObject> {
  
  std::vector<glm::vec3> pos;
  std::vector<glm::vec3> vel;

  size_t size() const {
    return pos.size();
  }
};

}
