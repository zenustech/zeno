#pragma once


#include <vector>
#include <z2/ztd/vec.h>


namespace z2::ds {


struct Mesh {
    // points
    std::vector<ztd::vec3f> vert;
    std::vector<ztd::vec2f> uv_vert;

    // face corners
    std::vector<int> loop;
    std::vector<int> uv_loop;

    // faces
    std::vector<ztd::vec2i> poly;
};



}
