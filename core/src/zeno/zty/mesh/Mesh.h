#pragma once


#include <vector>
#include <zeno/math/vec.h>


ZENO_NAMESPACE_BEGIN
namespace zty {


struct Mesh {
    // points
    std::vector<math::vec3f> vert;

    // corners
    std::vector<uint32_t> loop;
    std::vector<math::vec2f> loop_uv;

    // faces
    std::vector<uint32_t> poly;
};



}
ZENO_NAMESPACE_END
