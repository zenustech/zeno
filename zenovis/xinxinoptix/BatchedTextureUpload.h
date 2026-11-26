#pragma once
#include "CompletedTextureUpload.h"

namespace zeno { namespace cluster {
#if defined(_WIN32)
#if defined(DemandLoading_EXPORTS)
#define ZENO_TRANSFER_API __declspec(dllexport)
#else
#define ZENO_TRANSFER_API __declspec(dllimport)
#endif
#else
#define ZENO_TRANSFER_API
#endif

ZENO_TRANSFER_API bool beginBatchedTextureTransfer(void* owner,unsigned int page,CUstream stream,bool newStorage);
ZENO_TRANSFER_API void endBatchedTextureTransfer();
ZENO_TRANSFER_API bool capturingTextureTransfer();
// Capture a small batch on the existing prepared worker, then submit it inline
// before its phase event. No second worker can escape its submission budget.
ZENO_TRANSFER_API void beginPreparedTextureBatch(void* owner,CUstream stream);
ZENO_TRANSFER_API void endPreparedTextureBatch();
ZENO_TRANSFER_API bool capturingTextureUnmaps();
ZENO_TRANSFER_API void deferTextureTransferAction(std::function<void()> action);
ZENO_TRANSFER_API CUresult mapTextureArray(CUarrayMapInfo* maps,unsigned int count,CUstream stream);
ZENO_TRANSFER_API CUresult copyTextureTile(const CUDA_MEMCPY2D* copy,CUstream stream);
// Active only on the prepared-upload worker. Account for eviction unmaps as
// well as new tile maps when bounding its submitted binding commands.
ZENO_TRANSFER_API void beforeTextureMapping(CUstream stream);
ZENO_TRANSFER_API void afterTextureMapping(CUstream stream);
ZENO_TRANSFER_API bool batchedTextureTransferPending(void* owner,unsigned int page);
ZENO_TRANSFER_API unsigned int pendingBatchedTextureTransfers(void* owner);
ZENO_TRANSFER_API void drainBatchedTextureTransfers(void* owner);
ZENO_TRANSFER_API void releaseTextureMappingRecorder();
ZENO_TRANSFER_API CUstream textureUploadStream(void* owner,CUstream requestStream);
ZENO_TRANSFER_API void releaseTextureUploadStream(void* owner);
ZENO_TRANSFER_API bool beginBatchedTextureUnmaps(void* owner,CUstream stream);
ZENO_TRANSFER_API void endBatchedTextureUnmaps();
ZENO_TRANSFER_API void afterTextureUnmaps(std::function<void()> release);
ZENO_TRANSFER_API bool retainTextureStandby();
ZENO_TRANSFER_API bool recycleCompletedTextureTransfers();

// The request's per-page lock outlives this scope. Its staging allocation is
// released only by an action submitted after the queued copies.
struct ScopedTextureTransfer {
    bool active;
    ScopedTextureTransfer(void* owner,unsigned int page,CUstream stream,bool newStorage)
        :active(beginBatchedTextureTransfer(owner,page,stream,newStorage)){}
    ScopedTextureTransfer(const ScopedTextureTransfer&)=delete;
    ScopedTextureTransfer& operator=(const ScopedTextureTransfer&)=delete;
    ~ScopedTextureTransfer(){if(active)endBatchedTextureTransfer();}
};
struct ScopedTextureUnmaps {
    bool active;
    ScopedTextureUnmaps(void* owner,CUstream stream):active(beginBatchedTextureUnmaps(owner,stream)){}
    ScopedTextureUnmaps(const ScopedTextureUnmaps&)=delete;
    ScopedTextureUnmaps& operator=(const ScopedTextureUnmaps&)=delete;
    ~ScopedTextureUnmaps(){if(active)endBatchedTextureUnmaps();}
};
#undef ZENO_TRANSFER_API
}}
