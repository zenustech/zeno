#pragma once

namespace zeno_water_device {
constexpr unsigned GridSegments = 8;
constexpr unsigned GridVertices = 81;
constexpr unsigned MaxDepth = 12;
constexpr unsigned ObjectClusterBits = 18;
constexpr unsigned ObjectClusterSpan = 1u << ObjectClusterBits;
constexpr unsigned MaxSceneSurfaces = 64;
struct Patch {
    unsigned int x, y, depth, coarseEdges, surface;
};
struct Surface {
    const float4* field;
    float center[3], size[2];
    float period, amplitude, boundaryWidth;
    unsigned int resolution, ellipse, sbt, clusterIdBase;
    float objectToGeometry[12],geometryToObject[12];
    unsigned int flipWinding,transformed;
};
struct Launch {
    const Surface* surfaces;
    const Patch* patches;
    unsigned int surfaceCount, patchCount;
};
struct SceneLaunch {
    const Launch* objects;
    unsigned int objectCount;
};
}
