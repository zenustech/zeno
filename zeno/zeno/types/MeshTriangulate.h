#pragma once

#include <zeno/types/Mesh.h>
#include <zeno/math/vec.h>
#include <vector>

ZENO_NAMESPACE_BEGIN
namespace types {

std::vector<math::vec3f> meshToTriangles(Mesh const &mesh);

}
ZENO_NAMESPACE_END
