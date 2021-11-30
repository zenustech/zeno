#pragma once

#include <zeno/zty/mesh/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace zty {

void meshTriangulate(Mesh &mesh);
std::vector<math::vec3f> meshToTriangles(Mesh const &mesh);

}
ZENO_NAMESPACE_END
