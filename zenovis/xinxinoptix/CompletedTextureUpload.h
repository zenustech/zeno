#pragma once
#include <cuda.h>
#include <functional>

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
ZENO_UPLOAD_API void enqueueTextureUpload(CUstream stream, void* owner, unsigned int page, std::function<void()> publish);
ZENO_UPLOAD_API bool textureUploadPending(void* owner, unsigned int page);
ZENO_UPLOAD_API unsigned int pendingTextureUploads(void* owner);
ZENO_UPLOAD_API void drainTextureUploads(void* owner);

template<class Loader> inline void publishTextureUpload(CUstream stream, Loader* loader, unsigned int page, bool evictable, unsigned long long entry) {
    enqueueTextureUpload(stream, loader, page, [=] { loader->setPageTableEntry(page, evictable, entry); });
}
}} // namespace zeno::cluster
#undef ZENO_UPLOAD_API
