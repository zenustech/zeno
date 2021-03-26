#pragma once


#include <zen/zen.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <vector>

namespace zenbase {

struct MeshObject : zen::IObject {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;
};

}
