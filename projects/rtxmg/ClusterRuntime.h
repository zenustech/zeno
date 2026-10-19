#pragma once
#include "ClusterCache.h"
#include "ClusterDevice.h"
#include <map>
#include <memory>
#include <mutex>
#include <optix.h>

namespace zeno::cluster {
// Hold only after Runtime::update returns. Transfer/decode remain independent.
std::recursive_mutex &renderSubmissionMutex();
struct RuntimeSettings {
    uint64_t geometryBytes = 2ull << 30, clasBytes = 2ull << 30;
    float pixelError = 2.0f;
    uint32_t loadsPerFrame = 32;
};
struct RuntimeStats {
    uint64_t geometryBytes{}, clasBytes{}, loads{}, evictions{};
    uint64_t updates{}, plans{}, staticSkips{}, gasBuilds{}, iasBuilds{}, renderTriangles{};
    uint32_t residentGroups{}, renderClusters{}, pendingGroups{}, minLod{}, maxLod{};
    float effectivePixelError{};
    double updateMs{}, planningMs{}, loadingMs{}, accelerationMs{};
    double syncMs{}, pageReadMs{};
    uint64_t syncCalls{}, allocationCalls{}, freeCalls{};
    bool planningPending{};
    bool settled{};
    double gpuLodMs{}, gpuClasMs{}, gpuGasMs{}, gpuIasMs{}, gpuUploadMs{}, gpuSceneMs{}, hostStreamingMs{};
    uint64_t sceneVersion{}, cutVersion{}, deferredUpdates{};
    uint32_t sharedGases{};
    double sceneLatencyMs{};
    double backgroundPrepareMs{};
    double hostTaskMs{};
    double submissionWaitMs{}, backgroundSubmissionWaitMs{};
    uint64_t gasBytes{};
    uint32_t rebuiltGases{}, reusedGases{};
    uint64_t gasCompactions{};
};
// Read-only diagnostic snapshot of the exact published cut, for comparison
// against an ordinary triangle GAS. Never used by the render path.
struct ReferenceCut {
    uint32_t geometry{}, placement{};
    std::map<uint32_t, std::vector<uint32_t>> groups;
};
class Runtime {
  public:
    Runtime(const std::filesystem::path &manifest, RuntimeSettings settings = {});
    ~Runtime();
    Runtime(const Runtime &) = delete;
    // Builds and rendering share ordered compute stream 0. Interactive host
    // preparation has a context-bound worker; only update() publishes a version.
    bool update(OptixDeviceContext context, uint32_t sbtIndex, const std::array<float, 3> &eye,
                float projectionScale, bool interactive = false);
    OptixTraversableHandle handle() const;
    const zeno_cluster_device::Cluster *descriptors() const;
    RuntimeStats stats() const;
    std::vector<ReferenceCut> referenceCut() const;
    void validateCut() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace zeno::cluster
