#pragma once


#include <vector>
#include <zeno/math/vec.h>
#include <zeno/zycl/vector.h>


ZENO_NAMESPACE_BEGIN
namespace types {


struct Mesh {
    // points
    zycl::vector<math::vec3f> vert;

    // face corners
    zycl::vector<int> loop;
    zycl::vector<math::vec2f> loop_uv;

    // faces
    zycl::vector<math::vec2i> poly;
};



}
ZENO_NAMESPACE_END
