#pragma once


#include <vector>
#include <zeno2/ztd/vec.h>


namespace zeno2::ds {


struct Mesh {
    // points
    std::vector<ztd::vec3f> vert;

    // face corners
    std::vector<int> loop;
    std::vector<ztd::vec2f> loop_uv;

    // faces
    struct MPoly {
        int start;
        int num;
    };
    std::vector<MPoly> poly;
};



}
