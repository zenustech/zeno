#pragma once

#include <zen/zen.h>
#include <zen/vec.h>
#include <vector>
#include <array>

struct OctreeObject : zen::IObject {  // should OctreeObject : PrimitiveObject?

    std::vector<std::array<int, 8>> children;
    std::vector<zen::vec3f> CoM;
    std::vector<float> mass;

    zen::vec3f offset;
    float scale;

};
