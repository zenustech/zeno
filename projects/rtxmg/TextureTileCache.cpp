#include "TextureTileCache.h"
#include <zstd.h>
#include <common/xxhash.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zeno::cluster::texturecache {
namespace {
using Clock=std::chrono::steady_clock;
constexpr uint64_t Magic=0x31454c4954474d5aull, Missing=~uint64_t(0);
struct Header {
    uint64_t magic{};
    uint32_t version{},headerBytes{};
    uint64_t sourcePath{},sourceBytes{},sourceStamp{};
    uint32_t format{},width{},height{},levels{},side{},blockBytes{};
    uint64_t tiles{},indexHash{};
};
struct Level { uint64_t first{};uint32_t x{},y{}; };
struct Tile { uint64_t offset{};uint32_t bytes{},codec{};uint64_t checksum{}; };
static_assert(sizeof(Header)==80 && sizeof(Level)==16 && sizeof(Tile)==24);
uint64_t hash(const void* bytes,size_t n) {
    return XXH64(bytes,n,0);
}
uint64_t pathHash(const std::filesystem::path& p) {
    const auto text=std::filesystem::absolute(p).lexically_normal().generic_u8string();
    return hash(text.data(),text.size());
}
uint64_t stamp(const std::filesystem::path& p) {return uint64_t(std::filesystem::last_write_time(p).time_since_epoch().count());}
template<class T>T at(const char* p){T value;std::memcpy(&value,p,sizeof(value));return value;}
uint64_t us(Clock::time_point start){return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count());}
struct KtxLevel {uint64_t offset,bytes,decoded;};
struct Ktx {
    Source info;
    uint32_t scheme{};
    uint64_t bytes{};
    std::vector<KtxLevel> levels;
    explicit Ktx(const std::filesystem::path& path) {
        std::ifstream in(path,std::ios::binary);char h[80]{};in.read(h,sizeof(h));
        const unsigned char magic[]={0xab,0x4b,0x54,0x58,0x20,0x32,0x30,0xbb,0x0d,0x0a,0x1a,0x0a};
        if(!in || std::memcmp(h,magic,12))throw std::runtime_error("Expected KTX2 source");
        info={at<uint32_t>(h+12),at<uint32_t>(h+20),at<uint32_t>(h+24),at<uint32_t>(h+40),0};
        info.blockBytes=info.format==139?8:(info.format==141||info.format==145||info.format==146)?16:0;
        scheme=at<uint32_t>(h+44);bytes=std::filesystem::file_size(path);
        if(!info.blockBytes || !info.width || !info.height || info.width>65536 || info.height>65536 ||
           !info.levels || info.levels>17 || (scheme!=0&&scheme!=2) || at<uint32_t>(h+28)!=0 ||
           at<uint32_t>(h+32)>1 || at<uint32_t>(h+36)!=1)throw std::runtime_error("Unsupported KTX2 source");
        for(uint32_t l=0;l<info.levels;++l) {
            char data[24];in.read(data,sizeof(data));
            if(!in)throw std::runtime_error("Truncated KTX2 level index");
            KtxLevel level{at<uint64_t>(data),at<uint64_t>(data+8),at<uint64_t>(data+16)};
            const auto expected=uint64_t((std::max(1u,info.width>>l)+3)/4)*((std::max(1u,info.height>>l)+3)/4)*info.blockBytes;
            if(level.decoded!=expected || !level.bytes || level.offset>bytes || level.bytes>bytes-level.offset ||
               (scheme==0&&level.bytes!=expected))throw std::runtime_error("Invalid KTX2 mip range");
            levels.push_back(level);
        }
    }
};
std::vector<char> indexBytes(const std::vector<Level>& levels,const std::vector<Tile>& tiles) {
    std::vector<char> data(levels.size()*sizeof(Level)+tiles.size()*sizeof(Tile));
    std::memcpy(data.data(),levels.data(),levels.size()*sizeof(Level));
    if(!tiles.empty())std::memcpy(data.data()+levels.size()*sizeof(Level),tiles.data(),tiles.size()*sizeof(Tile));
    return data;
}
} // namespace
Source inspect(const std::filesystem::path& source){return Ktx(source).info;}
std::filesystem::path cachePath(const std::filesystem::path& root,const std::filesystem::path& source) {
    const auto key=pathHash(source);std::string name(16,'0');
    for(size_t i=0;i<16;++i)name[15-i]="0123456789abcdef"[(key>>(4*i))&15];
    return root/(name+".ztc");
}
uint64_t estimate(const Source& s,const std::vector<uint32_t>& selected) {
    std::set<uint32_t> unique(selected.begin(),selected.end());uint64_t n=0;
    for(auto l:unique) {
        if(l>=s.levels)throw std::runtime_error("Texture cache mip index out of range");
        n+=uint64_t((std::max(1u,s.width>>l)+TileSide-1)/TileSide)*((std::max(1u,s.height>>l)+TileSide-1)/TileSide);
    }
    return sizeof(Header)+s.levels*sizeof(Level)+n*(sizeof(Tile)+uint64_t(TileSide/4)*(TileSide/4)*s.blockBytes);
}
BakeStats bake(const std::filesystem::path& source,const std::filesystem::path& root,const std::vector<uint32_t>& selected) {
    const Ktx ktx(source);const auto& s=ktx.info;
    std::set<uint32_t> chosen(selected.begin(),selected.end());
    std::vector<Level> levels(s.levels);std::vector<Tile> tiles;
    for(uint32_t l=0;l<s.levels;++l) {
        auto& level=levels[l];level={Missing,(std::max(1u,s.width>>l)+TileSide-1)/TileSide,(std::max(1u,s.height>>l)+TileSide-1)/TileSide};
        if(chosen.count(l)){level.first=tiles.size();tiles.resize(tiles.size()+uint64_t(level.x)*level.y);}
    }
    if(chosen.empty() || *chosen.rbegin()>=s.levels)throw std::runtime_error("Invalid texture cache mip selection");
    Header header{Magic,1,sizeof(Header),pathHash(source),ktx.bytes,stamp(source),s.format,s.width,s.height,s.levels,TileSide,s.blockBytes,tiles.size(),0};
    std::filesystem::create_directories(root);
    const auto path=cachePath(root,source),temporary=std::filesystem::path(path.native()+std::filesystem::path(".tmp-"+std::to_string(Clock::now().time_since_epoch().count())).native());
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove(path,ec);}} cleanup{temporary};
    std::ofstream out(temporary,std::ios::binary|std::ios::trunc);std::ifstream in(source,std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header),sizeof(header));auto index=indexBytes(levels,tiles);out.write(index.data(),index.size());
    BakeStats stats{};
    const size_t side=TileSide/4,tileBytes=side*side*s.blockBytes;
    std::vector<char> tile(tileBytes),encoded(ZSTD_compressBound(tileBytes));
    for(auto l:chosen) {
        const auto mip=ktx.levels[l];std::vector<char> packed(size_t(mip.bytes)),decoded(size_t(mip.decoded));
        in.seekg(mip.offset);in.read(packed.data(),packed.size());
        if(!in)throw std::runtime_error("KTX cache bake read failed");
        if(ktx.scheme==0)decoded=std::move(packed);
        else {const auto n=ZSTD_decompress(decoded.data(),decoded.size(),packed.data(),packed.size());if(ZSTD_isError(n)||n!=decoded.size())throw std::runtime_error("KTX cache bake decode failed");}
        stats.sourceBytes+=mip.bytes;stats.decodedBytes+=mip.decoded;
        const uint32_t width=(std::max(1u,s.width>>l)+3)/4,height=(std::max(1u,s.height>>l)+3)/4;
        for(uint32_t y=0;y<levels[l].y;++y)for(uint32_t x=0;x<levels[l].x;++x) {
            std::fill(tile.begin(),tile.end(),0);
            for(size_t row=0;row<std::min<size_t>(side,height-y*side);++row)
                std::memcpy(tile.data()+row*side*s.blockBytes,decoded.data()+((y*side+row)*width+x*side)*s.blockBytes,
                            std::min<size_t>(side,width-x*side)*s.blockBytes);
            const auto n=ZSTD_compress(encoded.data(),encoded.size(),tile.data(),tile.size(),1);
            if(ZSTD_isError(n))throw std::runtime_error("Texture tile compression failed");
            auto& record=tiles[size_t(levels[l].first)+size_t(y)*levels[l].x+x];
            record={uint64_t(out.tellp()),uint32_t(std::min(n,tile.size())),n<tile.size()?1u:0u,hash(tile.data(),tile.size())};
            out.write(record.codec?encoded.data():tile.data(),record.bytes);++stats.tiles;
        }
    }
    stats.fileBytes=uint64_t(out.tellp());index=indexBytes(levels,tiles);header.indexHash=hash(index.data(),index.size());
    if(std::filesystem::file_size(source)!=header.sourceBytes || stamp(source)!=header.sourceStamp)
        throw std::runtime_error("KTX source changed while building its texture cache");
    out.seekp(0);out.write(reinterpret_cast<const char*>(&header),sizeof(header));out.write(index.data(),index.size());out.flush();
    if(!out)throw std::runtime_error("Texture cache write failed");out.close();
#ifdef _WIN32
    if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))throw std::runtime_error("Texture cache publication failed");
#else
    std::filesystem::rename(temporary,path);
#endif
    return stats;
}
struct Reader::Impl {
    Header header{};std::vector<Level> levels;std::vector<Tile> tiles;
    std::filesystem::path path;
    uint64_t fileStamp{},fileBytes{};
};
Reader::Reader(std::unique_ptr<Impl> p):impl(std::move(p)){}
Reader::~Reader()=default;
std::unique_ptr<Reader> Reader::open(const std::filesystem::path& root,const std::filesystem::path& source,const Source& s) {
    if(root.empty())return {};
    try {
        auto p=std::make_unique<Impl>();p->path=cachePath(root,source);std::ifstream stream(p->path,std::ios::binary);
        if(!stream)return {};
        auto& h=p->header;stream.read(reinterpret_cast<char*>(&h),sizeof(h));
        const auto bytes=std::filesystem::file_size(cachePath(root,source));
        p->fileStamp=stamp(p->path);p->fileBytes=bytes;
        if(!stream || h.magic!=Magic || h.version!=1 || h.headerBytes!=sizeof(h) || h.side!=TileSide ||
           h.sourcePath!=pathHash(source) || h.sourceBytes!=std::filesystem::file_size(source) || h.sourceStamp!=stamp(source) ||
           h.format!=s.format || h.width!=s.width || h.height!=s.height || h.levels!=s.levels || h.blockBytes!=s.blockBytes ||
           !h.levels || h.levels>17 || h.tiles>(bytes-sizeof(h))/sizeof(Tile))return {};
        uint64_t maxTiles=0;
        for(uint32_t l=0;l<h.levels;++l)maxTiles+=uint64_t((std::max(1u,h.width>>l)+TileSide-1)/TileSide)*((std::max(1u,h.height>>l)+TileSide-1)/TileSide);
        if(h.tiles>maxTiles)return {};
        const auto indexSize=uint64_t(h.levels)*sizeof(Level)+h.tiles*sizeof(Tile);
        if(indexSize>bytes-sizeof(h))return {};
        std::vector<char> index(static_cast<size_t>(indexSize));stream.read(index.data(),index.size());
        if(!stream || hash(index.data(),index.size())!=h.indexHash)return {};
        p->levels.resize(h.levels);p->tiles.resize(size_t(h.tiles));
        std::memcpy(p->levels.data(),index.data(),h.levels*sizeof(Level));
        std::memcpy(p->tiles.data(),index.data()+h.levels*sizeof(Level),p->tiles.size()*sizeof(Tile));
        uint64_t expected=0;
        for(uint32_t l=0;l<h.levels;++l) {
            const auto v=p->levels[l];
            if(v.x!=(std::max(1u,h.width>>l)+TileSide-1)/TileSide || v.y!=(std::max(1u,h.height>>l)+TileSide-1)/TileSide)return {};
            if(v.first==Missing)continue;
            if(v.first!=expected)return {};
            expected+=uint64_t(v.x)*v.y;if(expected>h.tiles)return {};
        }
        if(expected!=h.tiles)return {};
        const auto raw=uint64_t(TileSide/4)*(TileSide/4)*h.blockBytes;
        uint64_t previousEnd=sizeof(h)+indexSize;
        for(const auto& t:p->tiles) {
            if(!t.bytes || t.bytes>raw || t.codec>1 || (!t.codec&&t.bytes!=raw) || t.offset!=previousEnd ||
               t.offset>bytes || t.bytes>bytes-t.offset)return {};
            previousEnd=t.offset+t.bytes;
        }
        if(previousEnd!=bytes)return {};
        return std::unique_ptr<Reader>(new Reader(std::move(p)));
    } catch(const std::filesystem::filesystem_error&) {return {};}
}
bool Reader::contains(uint32_t l) const{return l<impl->levels.size() && impl->levels[l].first!=Missing;}
uint32_t Reader::tileBytes() const{return (TileSide/4)*(TileSide/4)*impl->header.blockBytes;}
ReadStats Reader::read(uint32_t l,uint32_t x,uint32_t y,char* output) const {
    if(!contains(l) || x>=impl->levels[l].x || y>=impl->levels[l].y)throw std::runtime_error("Texture tile index out of range");
    const auto level=impl->levels[l];const auto t=impl->tiles[size_t(level.first)+size_t(y)*level.x+x];
    std::vector<char> encoded(t.bytes);const auto begin=Clock::now();
    // A stream per texture exhausts the Windows CRT's stdio table before a
    // large scene finishes loading. Each worker retains at most eight files;
    // independent workers keep IO/decompression parallel without a global lock.
    struct File {std::ifstream stream;uint64_t tick{},stamp{},bytes{};};
    struct Files {std::map<std::filesystem::path,File> open;uint64_t tick{};};
    thread_local Files files;
    auto it=files.open.find(impl->path);
    if(it!=files.open.end() && (it->second.stamp!=impl->fileStamp || it->second.bytes!=impl->fileBytes)) {
        files.open.erase(it);it=files.open.end();
    }
    if(it==files.open.end()) {
        if(files.open.size()>=8) {
            auto oldest=std::min_element(files.open.begin(),files.open.end(),[](const auto& a,const auto& b){return a.second.tick<b.second.tick;});
            files.open.erase(oldest);
        }
        it=files.open.try_emplace(impl->path).first;it->second.stream.open(impl->path,std::ios::binary);
        it->second.stamp=impl->fileStamp;it->second.bytes=impl->fileBytes;
    }
    auto& input=it->second.stream;it->second.tick=++files.tick;
    input.clear();input.seekg(t.offset);input.read(encoded.data(),encoded.size());
    if(!input) {files.open.erase(it);throw std::runtime_error("Truncated texture tile payload");}
    ReadStats stats{t.bytes,us(begin),0};const auto decode=Clock::now();
    if(t.codec) {const auto n=ZSTD_decompress(output,tileBytes(),encoded.data(),encoded.size());if(ZSTD_isError(n)||n!=tileBytes())throw std::runtime_error("Texture tile decode failed");}
    else std::memcpy(output,encoded.data(),encoded.size());
    if(hash(output,tileBytes())!=t.checksum)throw std::runtime_error("Texture tile checksum mismatch");
    stats.decodeUs=us(decode);return stats;
}
} // namespace zeno::cluster::texturecache
