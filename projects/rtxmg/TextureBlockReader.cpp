#include "TextureBlockReader.h"
#include <zstd.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace zeno::cluster {
TextureBlockReadStats readTextureBlockTile(const std::filesystem::path& path,uint64_t offset,uint64_t encoded,uint64_t decoded,
    unsigned int scheme,unsigned int width,unsigned int height,unsigned int blockBytes,
    unsigned int tileX,unsigned int tileY,unsigned int tileWidth,unsigned int tileHeight,
    uint64_t windowBudget,char* destination) {
    TextureBlockReadStats stats;
    if(!tileWidth||!tileHeight||tileWidth%4||tileHeight%4)throw std::runtime_error("Invalid BC tile dimensions");
    const uint64_t bw=(width+3)/4,bh=(height+3)/4,tw=tileWidth/4,th=tileHeight/4;
    const uint64_t x=uint64_t(tileX)*tw,y=uint64_t(tileY)*th,rowBytes=tw*blockBytes;
    if(decoded!=bw*bh*blockBytes)throw std::runtime_error("Invalid streamed BC mip size");
    std::memset(destination,0,size_t(rowBytes*th));
    if(x>=bw||y>=bh)return stats;
    const uint64_t rows=std::min(th,bh-y),copyBytes=std::min(tw,bw-x)*blockBytes;
    std::ifstream input(path,std::ios::binary);input.seekg(offset);
    if(!input)throw std::runtime_error("Cannot open streamed texture: "+path.u8string());
    if(!scheme) {
        if(encoded!=decoded)throw std::runtime_error("Invalid uncompressed KTX mip");
        for(uint64_t row=0;row<rows;++row) {
            input.seekg(offset+((y+row)*bw+x)*blockBytes);
            input.read(destination+row*rowBytes,copyBytes);
            if(!input)throw std::runtime_error("KTX block-row read failed: "+path.u8string());
            stats.sourceBytes+=copyBytes;stats.decodedBytes+=copyBytes;
        }
        return stats;
    }
    if(scheme!=2)throw std::runtime_error("Unsupported streamed KTX supercompression");
    std::unique_ptr<ZSTD_DCtx,decltype(&ZSTD_freeDCtx)> context(ZSTD_createDCtx(),ZSTD_freeDCtx);
    if(!context)throw std::bad_alloc();
    unsigned int log=10;while(log<30 && (uint64_t(1)<<(log+1))<=windowBudget)++log;
    auto code=ZSTD_DCtx_setParameter(context.get(),ZSTD_d_windowLogMax,int(log));
    if(ZSTD_isError(code))throw std::runtime_error("Cannot bound the KTX Zstd window");
    std::array<char,128u<<10> source{},output{};
    ZSTD_inBuffer in{source.data(),0,0};uint64_t left=encoded,position=0;
    for(;;) {
        if(in.pos==in.size&&left) {
            const auto n=size_t(std::min<uint64_t>(left,source.size()));input.read(source.data(),n);
            if(!input)throw std::runtime_error("Streamed KTX input truncated: "+path.u8string());
            in={source.data(),n,0};left-=n;
            stats.sourceBytes+=n;
        }
        ZSTD_outBuffer out{output.data(),output.size(),0};const auto previous=in.pos;
        code=ZSTD_decompressStream(context.get(),&out,&in);
        if(ZSTD_isError(code))throw std::runtime_error("Streamed KTX Zstd decode failed (check source and host/window budget): "+path.u8string()+": "+ZSTD_getErrorName(code));
        if(position+out.pos>decoded)throw std::runtime_error("Streamed KTX output exceeds declared mip");
        const uint64_t sourceRow=bw*blockBytes;
        const auto beginRow=std::max(y,position/sourceRow),endRow=std::min(y+rows,(position+out.pos+sourceRow-1)/sourceRow);
        for(uint64_t actualRow=beginRow;actualRow<endRow;++actualRow) {
            const uint64_t row=actualRow-y,first=(actualRow*bw+x)*blockBytes,last=first+copyBytes;
            const auto a=std::max(first,position),b=std::min(last,position+out.pos);
            if(a<b)std::memcpy(destination+row*rowBytes+(a-first),output.data()+(a-position),size_t(b-a));
        }
        position+=out.pos;
        stats.decodedBytes+=out.pos;
        if(!code) {
            if(position!=decoded||left||in.pos!=in.size)throw std::runtime_error("Invalid streamed KTX frame length");
            return stats;
        }
        if(!left&&in.pos==in.size&&!out.pos&&in.pos==previous)throw std::runtime_error("Incomplete streamed KTX frame");
    }
}
}
