#pragma once
#include "WaterSurface.h"

namespace zeno::water {
struct SceneSurface {
    std::string key;
    Settings settings;
    bool visible=true;
};
struct PublishedSurface {
    std::string key;
    OptixTraversableHandle handle=0;
    uint64_t version=0;
};
// Independent bodies keep independent GAS, FFT state and publication fences.
// The device directory lets them continue sharing material hit programs.
class SceneRuntime {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    SceneRuntime();
    ~SceneRuntime();
    bool update(OptixDeviceContext context,uint32_t maxSbt,const std::vector<SceneSurface>& surfaces,View view,bool interactive);
    void rendered(cudaStream_t stream,bool auxiliary=false);
    zeno_water_device::SceneLaunch launch() const;
    const std::vector<PublishedSurface>& publications() const;
    Statistics statistics() const;
};
} // namespace zeno::water
