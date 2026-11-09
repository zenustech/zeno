#pragma once
#include <cuda.h>
#include <cstdlib>

namespace zeno { namespace cluster {
// A callback may change the worker's current context. Inspect it on each
// handoff; avoid an expensive redundant driver context reset when it matches.
inline CUresult setTextureContext(CUcontext context) {
    static const bool check=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_CONTEXT_GUARD");return !v||*v!='0';}();
    if(check) {
        CUcontext current{};const auto result=cuCtxGetCurrent(&current);
        if(result!=CUDA_SUCCESS)return result;
        if(current==context)return CUDA_SUCCESS;
    }
    return cuCtxSetCurrent(context);
}
// These SDK events protect buffer/page reuse; no caller measures elapsed time.
// Keep timestamps out of the stream dependency graph. The override is solely
// for controlled comparisons with the SDK's original event flags.
inline unsigned int textureSyncEventFlags() {
    static const bool timing=[] {
        const auto* value=std::getenv("ZENO_CLAS_TEXTURE_SYNC_EVENT_TIMING");
        return value&&*value=='1';
    }();
    return timing?CU_EVENT_DEFAULT:CU_EVENT_DISABLE_TIMING;
}
}}
