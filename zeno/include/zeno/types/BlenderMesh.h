#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <array>

using AttributeArray =
std::variant<std::vector<zeno::vec3f>, std::vector<float>>;

namespace zeno {

struct BlenderAxis : IObjectClone<BlenderAxis> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct BlenderMesh : IObjectClone<BlenderMesh, BlenderAxis> {
    bool is_smooth = false;

    std::vector<vec3f> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;

    std::map<std::string, AttributeArray> attrs;
};

}
