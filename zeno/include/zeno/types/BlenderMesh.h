#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>

struct BlenderMesh : zeno::IObjectClone<BlenderMesh> {
    std::vector<zeno::vec3f> vert;
    std::vector<std::tuple<int, int>> poly;
    std::vector<int> loop;
}
