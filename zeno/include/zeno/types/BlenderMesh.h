#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <array>

namespace zeno {

struct BlenderAxis : IObjectClone<BlenderAxis> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct BlenderMesh : IObjectClone<BlenderMesh, BlenderAxis> {
    bool is_smooth = false;

    std::vector<vec3f> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;
};

}
