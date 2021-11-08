#pragma once

#include <zeno/types/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace types {

void meshToTrianglesCPU(Mesh const &mesh, std::vector<math::vec3f> &vertices);
void meshToTriangles(Mesh const &mesh, zycl::vector<std::array<math::vec3f, 3>> &tris);

}
ZENO_NAMESPACE_END
