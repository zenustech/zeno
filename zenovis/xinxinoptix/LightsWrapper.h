#pragma once

#include "optixCommon.h"
#include "optixStuff.h"
#include "optixPathTracer.h"
#include <vector>

struct LightsWrapper {
    std::vector<float3> _planeLightGeo;
    std::vector<float4> _sphereLightGeo;
    std::vector<float3> _triangleLightGeo;
    std::vector<GenericLight> g_lights;

    std::vector<std::shared_ptr<OptixUtil::cuTexture>> texs;

    OptixTraversableHandle   lightIasHandle{};
    xinxinoptix::raii<CUdeviceptr>  lightIasBuffer{};

    OptixTraversableHandle   lightPlanesGas{};
    xinxinoptix::raii<CUdeviceptr>  lightPlanesGasBuffer{};

    OptixTraversableHandle  lightSpheresGas{};
    xinxinoptix::raii<CUdeviceptr> lightSpheresGasBuffer{};

    OptixTraversableHandle  lightTrianglesGas{};
    xinxinoptix::raii<CUdeviceptr> lightTrianglesGasBuffer{};

    xinxinoptix::raii<CUdeviceptr> lightBitTrailsPtr;
    xinxinoptix::raii<CUdeviceptr> lightTreeNodesPtr;
    xinxinoptix::raii<CUdeviceptr> lightTreeDummyPtr;

    xinxinoptix::raii<CUdeviceptr> triangleLightCoords;
    xinxinoptix::raii<CUdeviceptr> triangleLightNormals;

    void reset() { *this = {}; }
};