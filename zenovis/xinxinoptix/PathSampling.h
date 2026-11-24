#pragma once
#include "PathSamplingLaunch.h"

namespace zeno_path_sampling {
__forceinline__ __device__ unsigned hash(unsigned x) {
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);
}
__forceinline__ __device__ unsigned reverse(unsigned x) { return __brev(x); }
__forceinline__ __device__ float unit(unsigned x) { return float(x>>8)*0x1p-24f; }
__forceinline__ __device__ unsigned owen(unsigned x,unsigned seed) {
    // Fast nested binary permutation, as in SDK/cuda/random.h and Burley 2020.
    // Applied to the index it shuffles subnets; applied to a value it scrambles
    // its digits. These use independent keys below.
    x=reverse(x);
    x^=x*0x3d20adeau;x+=seed;x*=(seed>>16)|1u;
    x^=x*0x05526c56u;x^=x*0x53a22864u;
    return reverse(x);
}
__forceinline__ __device__ unsigned pascal(unsigned x) {
    // Second base-two digital-sequence coordinate (SDK/cuda/random.h::P).
    // Together with bit reversal this supplies a table-free 2D sequence.
    x^=x<<16;x^=(x&0x00ff00ffu)<<8;x^=(x&0x0f0f0f0fu)<<4;
    x^=(x&0x33333333u)<<2;x^=(x&0x55555555u)<<1;
    return x;
}
__forceinline__ __device__ unsigned legacyKey(const Launch& p,unsigned pixel,unsigned dimension) {
    // No sample-index bits enter the key. Independent shuffles per 2D block
    // decorrelate different uses while keeping each block's stratification.
    return hash(pixel^hash((dimension>>1)+p.seed+0x9e3779b9u));
}
__forceinline__ __device__ float legacySample(const Launch& p,unsigned pixel,unsigned index,unsigned dimension) {
    const unsigned key=legacyKey(p,pixel,dimension);
    const unsigned x=reverse(owen(index,hash(key^0x68bc21ebu)));
    const unsigned value=(dimension&1u)?pascal(x):x;
    return unit(owen(value,hash(key+((dimension&1u)?0x967a889bu:0x02e5be93u))));
}
__forceinline__ __device__ float2 legacySample2(const Launch& p,unsigned pixel,unsigned index,unsigned dimension) {
    const unsigned key=legacyKey(p,pixel,dimension);
    const unsigned x=reverse(owen(index,hash(key^0x68bc21ebu)));
    return make_float2(unit(owen(x,hash(key+0x02e5be93u))),
                       unit(owen(pascal(x),hash(key+0x967a889bu))));
}
__forceinline__ __device__ unsigned cycleOffset(unsigned cycle,unsigned bits) {
    // A bijection of the tile's pixels: every cycle uses a different scramble
    // before the mask repeats.
    const unsigned mask=(1u<<(2*bits))-1u;
    return reverse((cycle*0x9e3779b9u)&mask)>>(32-2*bits);
}
__forceinline__ __device__ unsigned pixelOffset(unsigned pixel,unsigned width,unsigned dimension,unsigned cycle,unsigned seed,bool independentTiles) {
    constexpr unsigned bits=7;
    constexpr unsigned mask=MaskSide-1u;
    unsigned x=pixel%width,y=pixel/width,tile=0;
    if(independentTiles) {
        // A fixed isometry and toroidal phase per spatial tile break visible
        // screen-wide repetition. No per-pixel shuffle: paired channels, each
        // tile's histogram and each pixel's temporal sequence stay intact.
        tile=hash((x>>bits)^hash((y>>bits)+seed));
        x&=mask;y&=mask;
        if(tile&1u){const unsigned t=x;x=y;y=t;}
        if(tile&2u)x=mask-x;
        if(tile&4u)y=mask-y;
    }
    // Move the entire multidimensional sample tuple together. Keeping relative
    // dimension offsets fixed avoids changing cross-dimension correlations
    // from one spatial tile to another.
    const unsigned offset=cycleOffset(cycle,bits),shift=hash(seed+dimension*0x9e3779b9u);
    x=(x+(shift&65535u)+(offset&mask)+((tile>>3)&mask))&mask;
    y=(y+(shift>>16)+(offset>>bits)+((tile>>(bits+3))&mask))&mask;
    return x+y*MaskSide;
}
__forceinline__ __device__ unsigned maskPosition(const Launch& p,unsigned pixel,unsigned width,unsigned index,unsigned pair) {
    constexpr unsigned bits=5;
    const unsigned xy=pixelOffset(pixel,width,pair,index>>bits,p.seed,p.independentTiles!=0);
    return (index&(MaskDepth-1u))*MaskSide*MaskSide+xy;
}
__forceinline__ __device__ float maskValue(unsigned char value,unsigned seed) {
    // Preserve the source's eight-bit strata while avoiding a discrete PDF
    // (and the exact-one endpoint) after CDF/BSDF transformations.
    return fminf((float(value)+unit(hash(seed)))*(1.f/256.f),0x1.fffffep-1f);
}
__forceinline__ __device__ float sample(const Launch& p,unsigned pixel,unsigned width,unsigned index,unsigned dimension) {
    if(p.mode==Legacy)return legacySample(p,pixel,index,dimension);
    const auto at=maskPosition(p,pixel,width,index,dimension);
    return maskValue(p.scalar[at],hash(pixel+dimension*0x9e3779b9u)^index);
}
__forceinline__ __device__ float2 sample2(const Launch& p,unsigned pixel,unsigned width,unsigned index,unsigned dimension) {
    if(p.mode==Legacy)return legacySample2(p,pixel,index,dimension);
    const auto at=maskPosition(p,pixel,width,index,dimension/2);
    const auto seed=hash(pixel+dimension*0x9e3779b9u)^index;
    return make_float2(maskValue(p.vector2[at*2],seed),maskValue(p.vector2[at*2+1],seed^0xb5297a4du));
}
}
