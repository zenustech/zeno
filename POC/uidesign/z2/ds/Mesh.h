#pragma once


#include <vector>
#include <z2/ztd/vec.h>


namespace z2::ds {


struct Mesh {
    // points
    std::vector<ztd::vec3f> vert;

    // face corners
    std::vector<int> loop;
    std::vector<ztd::vec2f> loop_uv;

    // faces
    std::vector<ztd::vec2i> poly;
};



}
