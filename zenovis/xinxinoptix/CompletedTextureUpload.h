#pragma once
#include <cuda.h>
#include <functional>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#if defined(DemandLoading_EXPORTS)
#define ZENO_UPLOAD_API __declspec(dllexport)
#else
#define ZENO_UPLOAD_API __declspec(dllimport)
#endif
#else
#define ZENO_UPLOAD_API
#endif

namespace zeno { namespace cluster {
// Exported from DemandLoading so renderer admission and OTK workers share one
// queue even when the SDK is a DLL. No CUDA host callback holds SDK locks.
ZENO_UPLOAD_API void enqueueTextureUpload(CUstream stream, void* owner, unsigned int page, std::function<void()> publish, bool tile);
ZENO_UPLOAD_API bool textureUploadPending(void* owner, unsigned int page);
ZENO_UPLOAD_API unsigned int pendingTextureUploads(void* owner);
// Commands already submitted to CUDA, excluding CPU-prepared transfer jobs.
ZENO_UPLOAD_API unsigned int submittedTextureUploads(void* owner);
ZENO_UPLOAD_API void drainTextureUploads(void* owner);
struct CompletedTextureStats {
    uint64_t tiles{},samplers{},eventsCreated{},eventsReused{},cachedEvents{},evictions{},recycledBuffers{};
    double evictionMs{},maxEvictionMs{};
};
ZENO_UPLOAD_API CompletedTextureStats completedTextureStats(void* owner);
// Exact sparse backing allocation; caller must pause admission and retire
// uploads first. Does not scan textures, query CUDA or synchronize the GPU.
ZENO_UPLOAD_API uint64_t textureTileBackingBytes(void* owner);
ZENO_UPLOAD_API void forgetTextureUploadStatistics(void* owner);
ZENO_UPLOAD_API void noteTextureEviction(void* owner,double milliseconds,unsigned int count=1);
ZENO_UPLOAD_API bool textureWorkDiagnosticsEnabled();
ZENO_UPLOAD_API void noteTexturePaging(unsigned int lists,unsigned int empty,unsigned int pages,
    unsigned int requested,unsigned int stale,bool evictionActive,double milliseconds,
    unsigned int restored,unsigned int invalidated,bool standby);
ZENO_UPLOAD_API void releaseTexturePagingRecorder();
ZENO_UPLOAD_API void noteTextureTransferRecycle(void* owner);
struct TextureCallbackStats {
    uint64_t queued{},active{},peak{},completed{};
    double oldestMs{},activeMs{};
};
// Read from the same DLL that owns the callback dispatcher.
ZENO_UPLOAD_API TextureCallbackStats textureCallbackStats();

struct PreparedTextureStats {
    unsigned int queued{}, inFlight{};
    uint64_t prepared{}, submitted{}, bytes{}, queuedBytes{};
    double submitMs{}, maxSubmitMs{}, readyLatencyMs{}, gpuMs{}, maxGpuMs{};
};
// CPU readers can run ahead of sparse mapping/copies. Only configured loaders
// use this queue; other OTK clients retain their normal request handling.
ZENO_UPLOAD_API void configurePreparedTextureUploads(void* owner);
ZENO_UPLOAD_API bool usePreparedTextureUploads(void* owner);
ZENO_UPLOAD_API bool executingPreparedTextureUpload(void* owner, unsigned int page);
ZENO_UPLOAD_API unsigned int preparedTextureReclaimPages(void* owner);
ZENO_UPLOAD_API bool preparedTextureUploadPending(void* owner, unsigned int page);
ZENO_UPLOAD_API bool enqueuePreparedTextureUpload(void* owner, unsigned int page, std::vector<char> data, std::function<void(CUstream)> upload,uint64_t resourceGroup=~uint64_t(0));
ZENO_UPLOAD_API bool copyPreparedTextureData(void* owner, unsigned int page, char* destination, size_t capacity);
ZENO_UPLOAD_API void updatePreparedTexturePriorities(void* owner,const uint64_t* keys,size_t count);
ZENO_UPLOAD_API void advancePreparedTextureUploads(void* owner, bool interactive, bool moving, double headroomMs=2.,bool geometryBusy=false,bool primaryPending=true);
ZENO_UPLOAD_API void pausePreparedTextureUploads(void* owner);
ZENO_UPLOAD_API void wakePreparedTextureUploads(void* owner);
ZENO_UPLOAD_API void drainPreparedTextureUploads(void* owner);
ZENO_UPLOAD_API void releasePreparedTextureUploads(void* owner);
ZENO_UPLOAD_API PreparedTextureStats preparedTextureStats(void* owner);

template<class Loader> inline void publishTextureUpload(CUstream stream, Loader* loader, unsigned int page, bool evictable, unsigned long long entry) {
    enqueueTextureUpload(stream, loader, page, [=] { loader->setPageTableEntry(page, evictable, entry); },evictable);
}
}} // namespace zeno::cluster
#undef ZENO_UPLOAD_API
