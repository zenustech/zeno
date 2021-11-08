#pragma once

#include <zeno/types/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace types {

void meshToTriangleVerticesCPU(Mesh const &mesh, std::vector<math::vec3f> &vertices);
void meshToTriangleVertices(Mesh const &mesh, zycl::vector<math::vec3f> &tris);
void meshToTriangleIndices(Mesh const &mesh, zycl::vector<math::vec3i> &tris);

}
ZENO_NAMESPACE_END
