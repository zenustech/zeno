#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <type_traits>
#include <string>
#include <cuda_runtime.h>
#include <optix_types.h>
#include "WaterDevice.h"
static_assert(std::is_trivial_v<zeno_water_device::Launch>,"OptiX constant launch parameters require a trivial water descriptor");

namespace zeno::water {
// A geometry-specific SBT variant of an ordinary bound material. The material
// ID and callable remain the same as those used by conventional primitives.
inline std::string shaderKey(const std::string& material) { return "__structured_water__/"+material; }
struct Settings {
    std::array<float,3> center{0,0,0};
    std::array<float,2> size{10,10}, wind{1,0};
    float amplitude=.008f, period=12.f, windSpeed=3.f, depth=1.f;
    float time=0, loopDuration=0, edgePixels=2.f, boundaryWidth=.1f;
    uint32_t resolution=256, seed=42, maxClusters=8192, ellipse=0;
    uint32_t sbt=0; // Resolved renderer binding; never part of the node descriptor.
    uint32_t clusterIdBase=0;
    std::array<float,12> objectToGeometry{1,0,0,0,0,1,0,0,0,0,1,0};
    std::array<float,12> poolToWorld{1,0,0,0,0,1,0,0,0,0,1,0};
    // Additional scene placements share one local-space GAS. Tessellation uses
    // the maximum demand of their world-space projections.
    std::vector<std::array<float,12>> instanceTransforms;
    bool operator==(const Settings& b) const;
    bool operator!=(const Settings& b) const { return !(*this==b); }
};
struct View {
    std::array<float,3> eye;
    float projection;
    std::array<float,3> right{1,0,0},up{0,1,0},front{0,0,-1};
    float tanX=0,tanY=0;
};
struct Statistics {
    uint32_t clusters=0, triangles=0, minDepth=0, maxDepth=0;
    uint64_t revision=0, topologyRevision=0, bytes=0;
    double submitMs=0;
    float fftMs=0, tessellationMs=0, clasMs=0, gasMs=0, effectiveEdgePixels=0;
    bool pending=false, qualityLimited=false, reusedTopology=false;
};
void validate(const Settings& settings);
class Runtime {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    Runtime();
    ~Runtime();
    Runtime(const Runtime&)=delete;
    Runtime& operator=(const Runtime&)=delete;
    // One pending complete surface at most. Interactive frames retain the
    // previous complete surface while its replacement builds on another stream.
    bool update(OptixDeviceContext context,uint32_t maxSbt,const std::vector<Settings>& surfaces,View view,bool interactive);
    void rendered(cudaStream_t stream,bool auxiliary=false);
    OptixTraversableHandle handle() const;
    zeno_water_device::Launch launch() const;
    Statistics statistics() const;
};
}
