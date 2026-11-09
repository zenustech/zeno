#pragma once
#include <cuda_runtime_api.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include "TextureDelivery.h"

namespace zeno::cluster {
constexpr unsigned int TextureFeedbackCounters=12;
// 0 physical pages, 1 retained, 2 speculative quota, 3 issued this frame,
// 4 useful, 5 expired unused, 6 issued total, 7 retention quota,
// 8 candidates, 9 primary misses, 10 active resident pages, 11 late.
// Only useful/expired/issued-total/late persist across feedback frames.
struct TextureFeedbackHints {
    const unsigned int* predicted=nullptr;
    unsigned int* recent=nullptr; // Four windows, sixteen completed frames each.
    unsigned int* issued=nullptr;
    unsigned int* counters=nullptr;
    unsigned int frame=0,firstTilePage=0,poolPages=0,retentionPages=0,prefetchPages=0,requestPages=0;
};
cudaError_t reduceTextureMisses(unsigned int* flags,unsigned int words,cudaStream_t stream);
cudaError_t reducePrimaryCoverage(const unsigned char* pixels,unsigned int count,unsigned int* counts,cudaStream_t stream);
// references is the SDK's per-launch bitset. primary has words+1 entries;
// its final word is scratch, used only on stream after rendering completes.
cudaError_t prioritizeTextureRequests(unsigned int* references,unsigned int* primary,
    const unsigned int* resident,unsigned int words,bool prioritize,cudaStream_t stream);
cudaError_t hintTextureRequests(unsigned int* references,unsigned int* primary,
    const unsigned int* resident,unsigned int words,bool prioritize,TextureFeedbackHints hints,cudaStream_t stream);
size_t textureImportanceScratchBytes();
cudaError_t rankTextureRequests(unsigned int* references,const unsigned int* primary,
    const unsigned int* resident,unsigned int words,unsigned int firstTilePage,unsigned int capacity,
    unsigned int pendingLimit,const zeno_cluster_device::TextureDemand* importance,
    unsigned long long* keysIn,unsigned long long* keysOut,unsigned int* counts,
    void* scratch,size_t scratchBytes,cudaStream_t stream);
}
