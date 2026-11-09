#pragma once

namespace clusterMaterial {
// Per-pixel history for interactive coarse texture fallbacks. Secondary-ray
// misses are deliberately excluded: stochastic paths must still converge at
// a fixed camera. Offline renders never bind this history buffer.
#if defined(__CUDACC__)
__host__ __device__
#endif
inline unsigned int textureHistory(unsigned int& state,unsigned int history,bool missing) {
    constexpr unsigned int countMask=0x00ffffffu;
    if(!history)state=0; // Camera, resolution, or explicit accumulation reset.
    unsigned int count=state&countMask,recovery=state>>24;
    count=count<history?count:history; // Respect geometry history limits too.
    if(missing)recovery=64;
    else if(recovery)--recovery;
    if(recovery && count>15)count=15;
    state=(count<countMask?count+1:countMask)|(recovery<<24);
    return count;
}
}
