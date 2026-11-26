#pragma once
#ifndef __CUDACC__
#include <cmath>
#endif
namespace zeno_cluster_device {
enum TextureSamplingFlags { CustomSampling=1,MinPoint=2,MagPoint=4,NoMipmaps=8,PointMipmaps=16 };
struct TextureSampleLevels {unsigned int first,second;float blend;bool point;};
#if defined(__CUDACC__)
__host__ __device__
#endif
inline TextureSampleLevels textureSampleLevels(unsigned int flags,float desired,unsigned int last) {
    const bool point=(flags&(desired<=0.f?MagPoint:MinPoint))!=0;
    float level=flags&NoMipmaps?0.f:fminf(fmaxf(desired,0.f),float(last));
    if(flags&PointMipmaps)level=floorf(level+.5f);
    const auto first=static_cast<unsigned int>(floorf(level));
    return {first,first<last?first+1:first,level-float(first),point};
}
#if defined(__CUDACC__)
__host__ __device__
#endif
inline float textureSampleCoordinate(float x,unsigned int pixels,bool point) {
    return point?(floorf(x*float(pixels))+.5f)/float(pixels):x;
}
}
