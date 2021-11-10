#pragma once

#include <zeno/types/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace types {

zycl::vector<math::vec3f> meshToTriangles(Mesh const &mesh);
zycl::vector<math::vec3f> meshToTrianglesCPU(Mesh const &mesh);

}
ZENO_NAMESPACE_END
