#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <array>
#include <vector>

namespace zeno {

struct PolyMesh {
    struct Polygon {
        int start = 0, len = 0;

        inline Polygon() = default;
        inline Polygon(int start, int len)
            : start(start), len(len) {}
    };

    std::vector<vec3f> vert;
    std::vector<Polygon> poly;
    std::vector<int> loop;
};

struct BlenderAxis : IObjectClone<BlenderAxis> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct BlenderMesh : IObjectClone<BlenderMesh, BlenderAxis>, PolyMesh {
    bool is_smooth = false;
};

}
