#pragma once

namespace zeno_water_device {
constexpr unsigned GridSegments = 8;
constexpr unsigned GridVertices = 81;
constexpr unsigned MaxDepth = 12;
struct Patch {
    unsigned int x, y, depth, coarseEdges, surface;
};
struct Surface {
    const float4* field;
    float center[3], size[2];
    float period, amplitude, roughness, ior;
    float absorption[3], boundaryWidth;
    unsigned int resolution, ellipse;
};
struct Launch {
    const Surface* surfaces;
    const Patch* patches;
    unsigned int surfaceCount, patchCount;
};
}
