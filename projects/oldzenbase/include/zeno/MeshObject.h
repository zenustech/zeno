#pragma once


#include <cstddef>
#include <memory>
#include <zeno/zeno.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <vector>

namespace zeno {

struct MeshObject : zeno::IObjectClone<MeshObject> {
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;
  size_t size()
  {
    return vertices.size();
  }
  void translate(const glm::vec3 &p)
  {
    #pragma omp parallel for
    for(int i=0;i<vertices.size();i++)
    {
      vertices[i] += p;
    }
  }
  std::shared_ptr<MeshObject> Clone()
  {
    std::shared_ptr<MeshObject> omesh = std::make_shared<MeshObject>();
    omesh->vertices.resize(vertices.size());
    omesh->uvs.resize(uvs.size());
    omesh->normals.resize(normals.size());
    #pragma omp parallel for
    for(int i=0;i<vertices.size();i++)
    {
      omesh->vertices[i] = vertices[i];
      omesh->uvs[i] = uvs[i];
      omesh->normals[i] = normals[i];
    }
    return omesh;
  }
};

}
