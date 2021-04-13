#if 0 // TODO: no more MATRIX OBJECT
#include <zen/zen.h>
#include <zen/MeshObject.h>
#include <zen/MatrixObject.h>
#include <glm/glm.hpp>
#include <cstring>

namespace zenbase {


static glm::vec3 mapplypos(glm::mat4 const &matrix, glm::vec3 const &vector) {
  auto vector4 = matrix * glm::vec4(vector, 1.0f);
  return glm::vec3(vector4) / vector4.w;
}


static glm::vec3 mapplydir(glm::mat4 const &matrix, glm::vec3 const &vector) {
  auto vector4 = matrix * glm::vec4(vector, 0.0f);
  return glm::vec3(vector4);
}


struct TransformMesh : zen::INode {
  virtual void apply() override {
    auto mesh = get_input("mesh")->as<MeshObject>();
    auto outmesh = zen::IObject::make<MeshObject>();
    auto matrix = get_input("matrix")->as<MatrixObject>()->to_4x4();
    for (auto const &x: mesh->vertices) {
      outmesh->vertices.push_back(mapplypos(matrix, x));
    }
    for (auto const &x: mesh->uvs) {
      outmesh->uvs.push_back(x);
    }
    for (auto const &x: mesh->normals) {
      outmesh->normals.push_back(glm::normalize(mapplydir(matrix, x)));
    }
    set_output("mesh", outmesh);
  }
};


static int defTransformMesh = zen::defNodeClass<TransformMesh>("TransformMesh",
    { /* inputs: */ {
    "mesh",
    "matrix",
    }, /* outputs: */ {
    "mesh",
    }, /* params: */ {
    }, /* category: */ {
    "trimesh",
    }});

}
#endif
