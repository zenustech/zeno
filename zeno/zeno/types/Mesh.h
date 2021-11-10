#pragma once


#include <vector>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace types {


struct Mesh {
    // points
    std::vector<math::vec3f> vert;

    // face corners
    std::vector<int> loop;
    std::vector<math::vec2f> loop_uv;

    // faces
    std::vector<math::vec2i> poly;
};



}
ZENO_NAMESPACE_END
