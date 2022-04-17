#pragma once

#include <zeno/zeno.h>
#include <zeno/vec.h>
#include <vector>
#include <array>

struct OctreeObject : zeno::IObject {  // should OctreeObject : PrimitiveObject?

    std::vector<std::array<int, 8>> children;
    std::vector<zeno::vec3f> CoM;
    std::vector<float> mass;

    zeno::vec3f offset;
    float radius;

};
