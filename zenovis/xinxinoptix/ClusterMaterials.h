#pragma once
#include "ClusterMaterialDevice.h"
#include "ClusterOpacity.h"
#include "TextureMotion.h"
#include "HostTextureCache.h"
#include <cstdint>
#include <filesystem>
#include <memory>

namespace zeno::cluster {
struct MaterialQueueStats {
    uint64_t batches{},unknownBatches{},cpuTasks{},uploads{},prepared{},feedbackContexts{};
    uint64_t ticketSlots{},completedTickets{};
    uint64_t callbacksQueued{},callbacksActive{},callbacksPeak{},callbacksCompleted{};
    uint64_t admissionLimit{},admissionPending{},admissionUnknown{},admissionAllowed{};
    double oldestBatchMs{},oldestCallbackMs{},activeCallbackMs{};
};
struct MaterialStats {
    uint64_t textures{},virtualBytes{},deviceBytes{},transferredBytes{},tilesRead{},evictions{},hostMipBytes{},hostPeakBytes{};
    uint64_t mipLoads{},sourceBytes{};
    double prepareMs{},requestMs{},statisticsMs{},sourceReadMs{},decodeMs{};
    unsigned int requestsAllowed{},pendingRequests{};
    uint64_t cacheTileLoads{},cacheTileBytes{},cacheFallbackTiles{},cacheTileErrors{};
    unsigned int preparedQueued{},uploadsInFlight{};
    uint64_t preparedTiles{},submittedTiles{},submittedBytes{},preparedBytes{};
    double submitCpuMs{},maxSubmitCpuMs{},readyToSubmitMs{},uploadGpuMs{},maxUploadGpuMs{};
    uint64_t readyTileMappings{},readySamplerMappings{};
    uint64_t decodedBytes{},deliveredBytes{},cacheHits{},cacheMisses{},cacheReloads{};
    uint32_t primaryPixels{},primaryMissingPixels{};
    uint64_t uploadEventsCreated{},uploadEventsReused{},uploadEventsCached{};
    uint32_t retainedPages{},prefetchIssued{},prefetchUseful{},prefetchUnused{},residentPages{};
    uint32_t prefetchCandidates{},primaryMissingPages{},activeResidentPages{},prefetchLate{};
    uint64_t liveTextureEvictions{};
    double evictionSubmitMs{},maxEvictionSubmitMs{};
    HostTileStats hostTiles{};
    HostBlockStats hostWorkspace{};
    uint64_t hostWorkspaceBudget{},hostTileBudget{};
    bool ramTiles=false;
    bool hostTileRetentionActive=false;
};
class Materials {
public:
    Materials(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB);
    ~Materials();
    Materials(const Materials&)=delete;
    // Close the previous upload window before scene/camera GPU preparation.
    void beginFrame();
    bool primaryDetailPending() const;
    // Prepare mappings before OptiX; collect requests after it. No host IO wait
    // in interactive mode. The renderer retries offline samples after misses.
    void prepare(bool interactive=false,uint32_t pixels=0,bool moving=false,bool geometryBusy=false);
    void beginPass();
    void endPass(bool interactive=false,bool moving=false);
    // Feed the completed viewport wall time back into interactive admission.
    // Uses the existing display completion, without another CUDA wait.
    void frameComplete(double milliseconds,bool geometryBusy=false,uint64_t buildGrowth=0);
    bool missed(bool wait=true);
    void wait();
    void clearResidency();
    // Diagnostic isolation only. Drain once, then retain the same mappings
    // and launch context without requesting or evicting texture pages.
    void setFrozen(bool frozen);
    void setView(const TextureView& view,bool interactive);
    bool settled() const;
    // Opt-in diagnostics: live work and age, separate from cumulative reads.
    MaterialQueueStats queueStats() const;
    zeno_cluster_device::MaterialLaunch launch(bool twoSidedOpaque=true,bool opaqueShadowAllowed=true) const;
    // Interactive diagnostics read cheap counters; only offline callers may
    // request a fresh SDK scan, which can contend with texture allocation.
    MaterialStats stats(bool wait=false) const;
    const OpaqueSlotPolicy& opaqueShadowPolicy() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
