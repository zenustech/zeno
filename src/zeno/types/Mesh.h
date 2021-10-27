#pragma once


#include <vector>
#include <zeno/ztd/vec.h>
#include <zeno/zycl/vector.h>


ZENO_NAMESPACE_BEGIN
namespace types {


struct Mesh {
    // points
    zycl::vector<ztd::vec3f> vert;

    // face corners
    zycl::vector<int> loop;
    zycl::vector<ztd::vec2f> loop_uv;

    // faces
    struct MPoly {
        int start{};
        int num{};
    };
    zycl::vector<MPoly> poly;
};



}
ZENO_NAMESPACE_END
