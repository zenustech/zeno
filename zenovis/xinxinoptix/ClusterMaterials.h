#pragma once
#include "ClusterMaterialDevice.h"
#include <cstdint>
#include <filesystem>
#include <memory>

namespace zeno::cluster {
struct MaterialStats {
    uint64_t textures{},virtualBytes{},deviceBytes{},transferredBytes{},tilesRead{},evictions{},hostMipBytes{},hostPeakBytes{};
    uint64_t mipLoads{},sourceBytes{};
    double prepareMs{},requestMs{},statisticsMs{},sourceReadMs{},decodeMs{};
    unsigned int requestsAllowed{},pendingRequests{};
};
class Materials {
public:
    Materials(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB);
    ~Materials();
    Materials(const Materials&)=delete;
    // Prepare mappings before OptiX; collect requests after it. No host IO wait
    // in interactive mode. The renderer retries offline samples after misses.
    void prepare();
    void beginPass();
    void endPass(bool interactive=false,bool moving=false);
    // Feed the completed viewport wall time back into interactive admission.
    // Uses the existing display completion, without another CUDA wait.
    void frameComplete(double milliseconds);
    bool missed(bool wait=true);
    void wait();
    bool settled() const;
    zeno_cluster_device::MaterialLaunch launch(bool twoSidedOpaque=true) const;
    // Interactive diagnostics read cheap counters; only offline callers may
    // request a fresh SDK scan, which can contend with texture allocation.
    MaterialStats stats(bool wait=false) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
