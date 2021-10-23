#pragma once


#include <vector>
#include <zs/ztd/vec.h>


namespace zs::zeno::ds {


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
