#pragma once

#include <zeno/core/IObject.h>
#include <zeno/utils/vec.h>
#include <array>
#include <vector>

namespace zeno {

struct PolyMesh {
    struct Polygon {
        int start = 0, len = 0;
    };

    std::vector<vec3f> vert;
    std::vector<Polygon> poly;
    std::vector<int> loop;
};

struct TransformObject : IObjectClone<TransformObject> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct PolyMeshObject : IObjectClone<PolyMeshObject, TransformObject>, PolyMesh {
    bool is_smooth = false;
};

}
