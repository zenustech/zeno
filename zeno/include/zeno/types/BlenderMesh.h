#pragma once

#include <zeno/core/IObject.h>
#include <zeno/types/AttrVector.h>
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

    AttrVector<vec3f> vert;
    AttrVector<Polygon> poly;
    AttrVector<int> loop;
};

struct BlenderAxis : IObjectClone<BlenderAxis> {
    std::array<std::array<float, 4>, 4> matrix;
};

struct BlenderMesh : IObjectClone<BlenderMesh, BlenderAxis>, PolyMesh {
    bool is_smooth = false;
};

using BlenderInputsType = std::map<std::string, std::function<std::shared_ptr<zeno::BlenderAxis>()>>;
using BlenderOutputsType = std::map<std::string, std::shared_ptr<zeno::BlenderAxis>>;

}
