#include "TextureFeedback.h"
#include "TextureResidency.h"
#include <OptiXToolkit/DemandLoading/Paging.h>
#include <cub/device/device_radix_sort.cuh>

namespace zeno::cluster {
namespace {
__global__ void initHints(unsigned int* primaryMissing,unsigned int* counts) {
    if(threadIdx.x==0)*primaryMissing=0;
    if(threadIdx.x<4||(threadIdx.x>=7&&threadIdx.x<=10))counts[threadIdx.x]=0;
}
__global__ void inspectHints(const unsigned int* references,const unsigned int* primary,const unsigned int* resident,unsigned int words,TextureFeedbackHints h) {
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=words)return;
    const auto missing=primary[i]&~resident[i];
    if(missing){atomicOr(const_cast<unsigned int*>(primary)+words,1u);atomicAdd(h.counters+9,__popc(missing));}
    if(i*32u>=h.firstTilePage) {
        const auto n=__popc(resident[i]);if(n)atomicAdd(h.counters,n);
        const auto active=__popc((references[i]|primary[i])&resident[i]);if(active)atomicAdd(h.counters+10,active);
    }
    if(h.issued) {
        const auto demanded=h.issued[i]&primary[i];
        const auto used=demanded&resident[i],late=demanded&~resident[i];
        if(used)atomicAdd(h.counters+4,__popc(used));
        if(late)atomicAdd(h.counters+11,__popc(late));
        auto left=h.issued[i]&~demanded;
        if(h.frame%64==0){if(left)atomicAdd(h.counters+5,__popc(left));left=0;}
        h.issued[i]=left;
    }
    if(h.predicted&&i*32u>=h.firstTilePage) {
        const auto candidate=h.predicted[i]&~(resident[i]|primary[i]|h.issued[i]);
        if(candidate)atomicAdd(h.counters+8,__popc(candidate));
    }
}
__device__ unsigned int takeBits(unsigned int bits,unsigned int count) {
    unsigned int result=0;
    while(bits&&count--){const auto bit=bits&(~bits+1u);result|=bit;bits&=bits-1;}
    return result;
}
__global__ void mergeHints(unsigned int* references,const unsigned int* primary,const unsigned int* resident,
                            unsigned int words,bool prioritize,TextureFeedbackHints h) {
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=words)return;
    const auto all=references[i]|primary[i];
    unsigned int recent=0;
    if(h.recent) {
        const unsigned int slot=(h.frame/16)%4;
        auto& current=h.recent[slot*words+i];
        current=(h.frame%16==0?0u:current)|(primary[i]&resident[i]);
        for(unsigned int k=0;k<4;++k)recent|=h.recent[k*words+i];
    }
    const bool missing=primary[words]!=0;
    const auto spare=h.requestPages>h.counters[9]?h.requestPages-h.counters[9]:0u;
    const auto quota=min(h.prefetchPages,spare);
    // A full physical LRU cache is normal. Stop speculation when the current
    // working set is hot, or current primary demand needs all admission slots.
    const bool prefetch=prioritize&&h.predicted&&quota&&h.counters[8]&&h.counters[10]*10ull<h.poolPages*9ull;
    unsigned int result=missing&&prioritize?primary[i]|(all&resident[i]):all;
    if(prefetch) {
        // Every primary miss fits first. Speculation uses only remaining
        // admission slots; resident secondary LRU references stay intact.
        result=primary[i]|(all&resident[i]);
        if(i*32u>=h.firstTilePage) {
            const auto candidate=h.predicted[i]&~(resident[i]|primary[i]|h.issued[i]);
            if(candidate) {
                const auto base=atomicAdd(h.counters+2,__popc(candidate));
                const auto chosen=base<quota?takeBits(candidate,quota-base):0u;
                if(chosen){result|=chosen;h.issued[i]|=chosen;const auto n=__popc(chosen);atomicAdd(h.counters+3,n);atomicAdd(h.counters+6,n);}
            }
        }
    }
    if(!missing&&h.retentionPages&&i*32u>=h.firstTilePage) {
        const auto candidate=recent&resident[i]&~all;
        if(candidate) {
            const auto base=atomicAdd(h.counters+7,__popc(candidate));
            const auto chosen=base<h.retentionPages?takeBits(candidate,h.retentionPages-base):0u;
            result|=chosen;if(chosen)atomicAdd(h.counters+1,__popc(chosen));
        }
    }
    references[i]=result;
}
__global__ void primaryCoverage(const unsigned char* pixels,unsigned int count,unsigned int* counts) {
    unsigned int touched=0,missing=0;
    for(unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        const auto flags=pixels[i];touched+=(flags&2u)!=0;missing+=(flags&3u)==3u;
    }
    for(unsigned int shift=16;shift;shift>>=1) {
        touched+=__shfl_down_sync(0xffffffffu,touched,shift);missing+=__shfl_down_sync(0xffffffffu,missing,shift);
    }
    if((threadIdx.x&31)==0){atomicAdd(counts,touched);atomicAdd(counts+1,missing);}
}
__global__ void reduceMisses(unsigned int* flags,unsigned int words) {
    unsigned int value=0;
    for(unsigned int i=threadIdx.x;i<words;i+=32)value|=flags[i];
    for(unsigned int shift=16;shift;shift>>=1)value|=__shfl_down_sync(0xffffffffu,value,shift);
    if(threadIdx.x==0)flags[words]=value;
}
__global__ void findPrimaryMisses(const unsigned int* primary,const unsigned int* resident,
                                  unsigned int words,unsigned int* missing) {
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    const bool miss=i<words && (primary[i]&~resident[i])!=0;
    if(__any_sync(0xffffffffu,miss) && (threadIdx.x&31)==0)atomicOr(missing,1u);
}
__global__ void mergeReferences(unsigned int* references,const unsigned int* primary,
                                const unsigned int* resident,unsigned int words,bool prioritize) {
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<words) {
        // Keep every resident secondary reference for LRU/eviction. Defer
        // only new secondary misses while primary requests need admission.
        // Nonpriority frames and primary-complete frames admit all requests.
        const auto all=references[i]|primary[i];
        references[i]=prioritize && primary[words] ? primary[i]|(all&resident[i]) : all;
    }
}

// Ranking is bounded by a fixed hash table and keeps all resident references.
// A cohort larger than the live admission bound prevents already-pending
// high-scoring pages from consuming every selected slot on subsequent frames.
__global__ void rankCounts(const unsigned int* primary,const unsigned int* resident,
                           unsigned int firstTilePage,unsigned int* counts) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i*32u<firstTilePage) {
        const auto missing=primary[i]&~resident[i];
        if(missing)atomicAdd(counts,__popc(missing));
    }
}
__global__ void rankKeys(const zeno_cluster_device::TextureDemand* hints,const unsigned int* primary,
                         const unsigned int* resident,unsigned int pages,unsigned int firstTilePage,
                         unsigned long long* keys,unsigned int* counts) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=zeno_cluster_device::TextureDemandSlots)return;
    const auto h=hints[i];unsigned long long key=0;
    if(h.page && h.page-1>=firstTilePage && h.page-1<pages) {
        const auto p=h.page-1;
        if(!demandLoading::checkBitSet(p,resident)) {
            const bool fine=demandLoading::checkBitSet(p,primary);
            const auto score=fine?max(h.fine,h.parent):h.parent;
            if(score) {
                key=(fine?(1ull<<63):0ull)|(static_cast<unsigned long long>(min(score,0x7fffffffu))<<32)|(0xffffffffu-p);
                atomicAdd(counts+(fine?1:2),1u);
            }
        }
    }
    keys[i]=key;
}
__global__ void rankClear(unsigned int* references,const unsigned int* primary,const unsigned int* resident,
                          unsigned int words,unsigned int firstTilePage,const unsigned int* counts) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=words)return;
    const auto all=references[i]|primary[i];
    references[i]=(!primary[words] || i*32u<firstTilePage || counts[1]+counts[2]==0)?all:(all&resident[i]);
}
__global__ void rankSelect(unsigned int* references,const unsigned long long* keys,const unsigned int* counts,
                           unsigned int capacity,unsigned int pendingLimit,const unsigned int* primary,unsigned int words) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=zeno_cluster_device::TextureDemandSlots)return;
    const auto k=keys[i];if(!k || !primary[words] || counts[0]>=capacity)return;
    const unsigned int remaining=capacity-counts[0];
    const unsigned int parentQuota=min(8u,remaining/4u);
    const bool fine=(k>>63)!=0;
    const auto fineLimit=pendingLimit+remaining-parentQuota;
    if(fine?i>=fineLimit:i-counts[1]>=parentQuota)return;
    const unsigned int page=0xffffffffu-static_cast<unsigned int>(k);
    atomicOr(references+(page>>5),1u<<(page&31));
}

}
cudaError_t reduceTextureMisses(unsigned int* flags,unsigned int words,cudaStream_t stream) {
    reduceMisses<<<1,32,0,stream>>>(flags,words);
    return cudaGetLastError();
}
cudaError_t reducePrimaryCoverage(const unsigned char* pixels,unsigned int count,unsigned int* counts,cudaStream_t stream) {
    auto status=cudaMemsetAsync(counts,0,2*sizeof(unsigned int),stream);
    if(status!=cudaSuccess||!count)return status;
    primaryCoverage<<<128,256,0,stream>>>(pixels,count,counts);return cudaGetLastError();
}
cudaError_t prioritizeTextureRequests(unsigned int* references,unsigned int* primary,
    const unsigned int* resident,unsigned int words,bool prioritize,cudaStream_t stream) {
    auto status=cudaMemsetAsync(primary+words,0,sizeof(unsigned int),stream);
    if(status!=cudaSuccess || !words)return status;
    if(prioritize)findPrimaryMisses<<<(words+255)/256,256,0,stream>>>(primary,resident,words,primary+words);
    mergeReferences<<<(words+255)/256,256,0,stream>>>(references,primary,resident,words,prioritize);
    return cudaGetLastError();
}
cudaError_t hintTextureRequests(unsigned int* references,unsigned int* primary,const unsigned int* resident,
    unsigned int words,bool prioritize,TextureFeedbackHints hints,cudaStream_t stream) {
    if(!words)return cudaSuccess;
    initHints<<<1,32,0,stream>>>(primary+words,hints.counters);
    inspectHints<<<(words+255)/256,256,0,stream>>>(references,primary,resident,words,hints);
    mergeHints<<<(words+255)/256,256,0,stream>>>(references,primary,resident,words,prioritize,hints);
    return cudaGetLastError();
}
size_t textureImportanceScratchBytes() {
    size_t bytes=0;
    cub::DeviceRadixSort::SortKeysDescending(nullptr,bytes,static_cast<unsigned long long*>(nullptr),
        static_cast<unsigned long long*>(nullptr),zeno_cluster_device::TextureDemandSlots);
    return bytes;
}
cudaError_t rankTextureRequests(unsigned int* references,const unsigned int* primary,const unsigned int* resident,
    unsigned int words,unsigned int firstTilePage,unsigned int capacity,unsigned int pendingLimit,
    const zeno_cluster_device::TextureDemand* importance,unsigned long long* keysIn,unsigned long long* keysOut,
    unsigned int* counts,void* scratch,size_t bytes,cudaStream_t stream) {
    auto e=cudaMemsetAsync(counts,0,3*sizeof(unsigned int),stream);if(e!=cudaSuccess)return e;
    rankCounts<<<(firstTilePage+8191)/8192,256,0,stream>>>(primary,resident,firstTilePage,counts);
    rankKeys<<<zeno_cluster_device::TextureDemandSlots/256,256,0,stream>>>(importance,primary,resident,words*32u,firstTilePage,keysIn,counts);
    e=cub::DeviceRadixSort::SortKeysDescending(scratch,bytes,keysIn,keysOut,zeno_cluster_device::TextureDemandSlots,0,64,stream);
    if(e!=cudaSuccess)return e;
    rankClear<<<(words+255)/256,256,0,stream>>>(references,primary,resident,words,firstTilePage,counts);
    rankSelect<<<zeno_cluster_device::TextureDemandSlots/256,256,0,stream>>>(references,keysOut,counts,capacity,pendingLimit,primary,words);
    return cudaGetLastError();
}
}
