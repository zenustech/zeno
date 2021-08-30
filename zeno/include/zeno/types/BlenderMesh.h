#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <array>

namespace zeno {

struct PolygonalMesh {
    std::vector<zeno::vec3f> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;
};

struct BlenderAxis : IObjectClone<BlenderAxis> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct BlenderMesh : IObjectClone<BlenderMesh, BlenderAxis>, PolygonalMesh {
    bool is_smooth = false;
};

}
