#pragma once


#include <vector>
#include <zeno/math/vec.h>
#include <zeno/types/field.h>


ZENO_NAMESPACE_BEGIN
namespace types {


struct Mesh {
    // points
    std::vector<math::vec3f> vert;

    // faces
    std::vector<std::vector<int>> poly;
};



}
ZENO_NAMESPACE_END
