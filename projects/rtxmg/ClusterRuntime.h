#pragma once
#include "ClusterCache.h"
#include "ClusterDevice.h"
#include "ClusterLod.h"
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
    bool materials = false;
    uint32_t textureMiB = 2048, hostTextureMiB = 256;
};
struct LodControl {
    bool adaptive = true;
    float pixelError = 0.f; // Zero uses the asset node's requested error.
};
struct RuntimeStats {
    uint64_t geometryBytes{}, clasBytes{}, loads{}, evictions{};
    uint64_t geometryReservedBytes{},clasReservedBytes{};
    uint64_t updates{}, plans{}, staticSkips{}, gasBuilds{}, iasBuilds{}, renderTriangles{};
    uint32_t residentGroups{}, renderClusters{}, pendingGroups{}, minLod{}, maxLod{};
    float effectivePixelError{};
    float requestedPixelError{}, poolLoad{}, smoothedPoolLoad{};
    bool adaptiveLod = true, qualityLimited{};
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
    uint64_t gasBytes{},gasReservedBytes{};
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
    // Interactive preparation owns the back scene on a context-bound worker
    // and a separate CUDA build stream. Only update() publishes a completed
    // version; last-use events protect front-scene storage during retirement.
    bool update(OptixDeviceContext context, uint32_t sbtIndex, const std::array<float, 3> &eye,
                float projectionScale, bool interactive = false, const ZoomView &zoom = {}, LodControl control = {});
    OptixTraversableHandle handle() const;
    const zeno_cluster_device::Cluster *descriptors() const;
    RuntimeStats stats() const;
    // Physical allocator counters, queried only at explicit resource snapshots.
    std::array<uint64_t,2> residencyMemory() const; // Reserved, used bytes.
    std::vector<ReferenceCut> referenceCut() const;
    void validateCut() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace zeno::cluster
