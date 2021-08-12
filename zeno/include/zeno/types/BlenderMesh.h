#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

namespace zeno {

struct BlenderMesh : IObjectClone<BlenderMesh> {
    std::vector<vec3f> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;
};

}
