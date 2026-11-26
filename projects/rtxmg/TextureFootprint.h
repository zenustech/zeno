#pragma once
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
#include <OptiXToolkit/DemandLoading/TileIndexing.h>
#ifndef __CUDACC__
#include <cmath>
using std::isfinite;
#endif

namespace zeno_cluster_device {
#if defined(__CUDACC__)
#define ZENO_FOOTPRINT_INLINE __host__ __device__ __forceinline__
#else
#define ZENO_FOOTPRINT_INLINE inline
#endif
struct TextureAxisSpan {unsigned int first=0,last=0;};
struct TextureAxisTiles {TextureAxisSpan spans[2];unsigned int count=0;};
ZENO_FOOTPRINT_INLINE float textureAddress(float x,unsigned int mode) {
    if(mode==CU_TR_ADDRESS_MODE_WRAP)x-=floorf(x);
    else if(mode==CU_TR_ADDRESS_MODE_MIRROR){x-=2.f*floorf(x*.5f);x=x<=1.f?x:2.f-x;}
    return fminf(fmaxf(x,0.f),0.999999940395355224609375f);
}
ZENO_FOOTPRINT_INLINE TextureAxisTiles textureAxisTiles(float center,float radius,unsigned int mode,
                                                        unsigned int pixels,unsigned int tile) {
    TextureAxisTiles out;
    if(!pixels||!tile||!isfinite(center)||!isfinite(radius))return out;
    radius=fmaxf(radius,0.f);float lo=center-radius,hi=center+radius;
    auto add=[&](float a,float b){
        auto& s=out.spans[out.count++];
        s.first=static_cast<unsigned int>(fminf(fmaxf(a,0.f),0.999999940395355224609375f)*pixels)/tile;
        s.last=static_cast<unsigned int>(fminf(fmaxf(b,0.f),0.999999940395355224609375f)*pixels)/tile;
    };
    if(mode==CU_TR_ADDRESS_MODE_WRAP) {
        if(hi-lo>=1.f)add(0.f,1.f);
        else if(floorf(lo)!=floorf(hi)){add(0.f,hi-floorf(hi));add(lo-floorf(lo),1.f);}
        else add(lo-floorf(lo),hi-floorf(hi));
    } else if(mode==CU_TR_ADDRESS_MODE_MIRROR) {
        if(hi-lo>=2.f)add(0.f,1.f);
        else {
            float a=textureAddress(lo,mode),b=textureAddress(hi,mode);
            float first=fminf(a,b),last=fmaxf(a,b),turn=ceilf(lo);
            if(turn<=hi){if(fmodf(fabsf(turn),2.f)==0.f)first=0.f;else last=1.f;}
            if(turn+1.f<=hi){first=0.f;last=1.f;}
            add(first,last);
        }
    } else {
        if(mode==CU_TR_ADDRESS_MODE_BORDER&&(hi<0.f||lo>1.f))return out;
        add(lo,hi);
    }
    // Wrapped intervals can touch the same tile, especially on a thin mip.
    // Visit that tile once so coverage scores count pages rather than spans.
    if(out.count==2&&out.spans[0].last+1>=out.spans[1].first) {
        out.spans[0].last=out.spans[1].last;out.count=1;
    }
    return out;
}
#if defined(__CUDACC__)
// Enumerate the complete rectangular filter support in both sampled levels.
// The callback can request, check residency, or attach an importance score.
template<class Visit> __device__ __forceinline__ bool visitTextureRect(
        const demandLoading::TextureSampler& s,float u,float v,float dx,float dy,
        unsigned int level,bool single,Visit visit) {
    bool result=true;
    for(unsigned int pass=0;pass<(single?1u:2u);++pass) {
        const unsigned int mip=level+pass;
        if(mip>=s.mipTailFirstLevel||mip>=demandLoading::MAX_TILE_LEVELS) {
            return visit(s.startPage)&&result;
        }
        const auto width=s.width>>mip,height=s.height>>mip;
        const auto xs=textureAxisTiles(u,dx,s.desc.wrapMode0,width?width:1u,1u<<s.desc.logTileWidth);
        const auto ys=textureAxisTiles(v,dy,s.desc.wrapMode1,height?height:1u,1u<<s.desc.logTileHeight);
        const auto sizes=s.mipLevelSizes[mip];
        for(unsigned int yi=0;yi<ys.count;++yi)for(unsigned int xi=0;xi<xs.count;++xi)
            for(unsigned int y=ys.spans[yi].first;y<=ys.spans[yi].last;++y)
                for(unsigned int x=xs.spans[xi].first;x<=xs.spans[xi].last;++x)
                    result=visit(s.startPage+sizes.mipLevelStart+y*sizes.levelWidthInTiles+x)&&result;
    }
    return result;
}
#endif
#undef ZENO_FOOTPRINT_INLINE
}
