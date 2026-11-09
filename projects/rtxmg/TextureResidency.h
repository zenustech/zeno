#pragma once
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#include <OptiXToolkit/DemandLoading/Paging.h>

namespace zeno_cluster_device {
#if defined(__CUDACC__)
#define ZENO_RESIDENCY_INLINE __host__ __device__ __forceinline__
#else
#define ZENO_RESIDENCY_INLINE inline
#endif
inline constexpr unsigned int InvalidTexturePage=0xffffffffu;
// A center-page lower bound is exact for the aligned power-of-two tile grids.
// Unsupported grids retain the established full-footprint fallback walk.
ZENO_RESIDENCY_INLINE bool residencyGridSupported(const demandLoading::TextureSampler& s) {
    return s.width && s.height && !(s.width&(s.width-1)) && !(s.height&(s.height-1)) &&
        s.desc.wrapMode0!=CU_TR_ADDRESS_MODE_MIRROR && s.desc.wrapMode1!=CU_TR_ADDRESS_MODE_MIRROR;
}
ZENO_RESIDENCY_INLINE unsigned int textureCenterPage(const demandLoading::TextureSampler& s,
                                                    float u,float v,unsigned int level) {
    if(!residencyGridSupported(s) || !isfinite(u) || !isfinite(v))return InvalidTexturePage;
    if(level>=s.mipTailFirstLevel)return s.startPage;
    if(level>=demandLoading::MAX_TILE_LEVELS)return InvalidTexturePage;
    u=demandLoading::wrapTexCoord(u,static_cast<CUaddress_mode>(s.desc.wrapMode0));
    v=demandLoading::wrapTexCoord(v,static_cast<CUaddress_mode>(s.desc.wrapMode1));
    const auto width=s.width>>level,height=s.height>>level;
    const auto x=static_cast<unsigned int>(u*(width?width:1))>>s.desc.logTileWidth;
    const auto y=static_cast<unsigned int>(v*(height?height:1))>>s.desc.logTileHeight;
    return s.startPage+s.mipLevelSizes[level].mipLevelStart+
        y*s.mipLevelSizes[level].levelWidthInTiles+x;
}
#if defined(__CUDACC__)
// Derive a MinMip lower bound only where a ray actually needs a fallback.
// A missing center page proves the complete footprint cannot be resident.
// The caller still validates that complete footprint and hardware residency.
__device__ __forceinline__ unsigned int residentMipLowerBound(const demandLoading::TextureSampler& s,
        const unsigned int* resident,float u,float v,unsigned int first,unsigned int last) {
    if(!residencyGridSupported(s) || !s.desc.isSparseTexture)return first;
    for(unsigned int level=first;level<=last;++level) {
        const auto page=textureCenterPage(s,u,v,level);
        if(page==InvalidTexturePage || demandLoading::checkBitSet(page,resident))return level;
    }
    return last+1;
}
#endif
#undef ZENO_RESIDENCY_INLINE
}
