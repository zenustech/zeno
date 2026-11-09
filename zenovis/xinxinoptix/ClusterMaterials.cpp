#include "ClusterMaterials.h"
#include "CompletedTextureUpload.h"
#include "ClusterProfiling.h"
#include "ClusterCache.h"
#include "TextureTileCache.h"
#include "TextureFeedback.h"
#include "TextureMaterialSet.h"
#include "TextureBudget.h"
#include <zeno/utils/FrameCapture.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/Ticket.h>
#include <OptiXToolkit/ImageSource/ImageSource.h>
#include <OptiXToolkit/ImageSource/TextureInfo.h>
#include <OptiXToolkit/Memory/Allocators.h>
#include <cuda_runtime.h>
#include <zstd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <list>
#include <set>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace zeno::cluster {
namespace {
using namespace zeno_cluster_device;
using Json=nlohmann::json;
void check(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
void check(CUresult r){if(r!=CUDA_SUCCESS){const char* message=nullptr;cuGetErrorString(r,&message);throw std::runtime_error(message?message:"CUDA driver error");}}
using Clock=std::chrono::steady_clock;
constexpr unsigned int MissingBuckets=128;
double elapsedMs(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct Buffer {
    void* p=nullptr;
    ~Buffer(){if(p)cudaFree(p);}
    template<class T>void upload(const std::vector<T>& v){check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
// Read-only CUDA memory snapshots stay off the render/input thread. No texture
// jobs, mappings or publication occur here. The sampler is bounded to 1 Hz.
class TextureMemorySampler {
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    bool stopping=false;
public:
    struct Sample {uint64_t free=0,total=0,sequence=0;};
    Sample latest;
    TextureMemorySampler() {
        CUcontext context{};check(cuCtxGetCurrent(&context));
        worker=std::thread([this,context] {
            if(cuCtxSetCurrent(context)!=CUDA_SUCCESS)return;
            for(;;) {
                size_t free=0,total=0;const auto status=cuMemGetInfo(&free,&total);
                std::unique_lock<std::mutex> lock(mutex);
                if(status==CUDA_SUCCESS)latest={free,total,latest.sequence+1};
                if(changed.wait_for(lock,std::chrono::seconds(1),[&]{return stopping;}))break;
            }
            cuCtxSetCurrent(nullptr);
        });
    }
    Sample sample(){std::lock_guard<std::mutex> lock(mutex);return latest;}
    ~TextureMemorySampler(){ {std::lock_guard<std::mutex> lock(mutex);stopping=true;}changed.notify_all();worker.join();}
};
// RAM tiles and the shared decoded workspace use one configured host budget.
// Source registration completes before OTK readers start; only payloads are
// allocated on demand. The original whole-mip path remains a comparison mode.
class MipCache {
    std::string traceFile;
    std::mutex traceMutex;
    std::map<std::string,std::set<uint32_t>> requestedLevels;
    uint32_t nextImage=0;
    std::mutex budgetMutex;
public:
    bool ramTiles;
    std::atomic<bool> retainTiles{false};
    uint64_t totalBudget,workspaceFloor;
    std::atomic<uint64_t> workspaceBudget,tileBudget;
    std::shared_ptr<HostCacheUsage> usage=std::make_shared<HostCacheUsage>();
    HostBlockCache blocks;
    std::unique_ptr<HostTileCache> tiles;
    std::atomic<uint64_t> mipLoads{0},sourceBytes{0},sourceReadUs{0},decodeUs{0};
    std::atomic<uint64_t> tileLoads{0},tileBytes{0},tileFallbacks{0},tileErrors{0};
    std::atomic<uint64_t> decodedBytes{0},deliveredBytes{0};
    using Lease=HostBlockCache::Lease;
    MipCache(uint64_t bytes,bool tileMode):ramTiles(tileMode),totalBudget(bytes),
        workspaceFloor(tileMode?std::min<uint64_t>(bytes/4,uint64_t(2)<<30):bytes),workspaceBudget(bytes),tileBudget(0),
        blocks(bytes,usage,framecapture::enabled()) {
        if(tileMode)tiles=std::make_unique<HostTileCache>(0,usage);
        if(const auto* p=std::getenv("ZENO_CLAS_TEXTURE_TRACE"))traceFile=p;
    }
    uint32_t registerImage(uint64_t mipBytes) {
        if(ramTiles) {
            workspaceFloor=std::max(workspaceFloor,std::min(totalBudget,mipBytes));
        }
        return ++nextImage;
    }
    void reserveTiles(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(budgetMutex);
        constexpr uint64_t Quantum=16ull<<20;
        const auto wanted=std::min(totalBudget-workspaceFloor,
            (tiles->stats().bytes+bytes+Quantum-1)/Quantum*Quantum);
        if(wanted<=tileBudget.load())return;
        // Shrink the workspace first; live leases may temporarily keep some
        // capacity. No lock is held across IO, decoding or a reader wait.
        const auto actual=blocks.trimCapacity(totalBudget-wanted);
        workspaceBudget=actual;tileBudget=totalBudget-actual;
        tiles->setCapacity(tileBudget.load());
    }
    bool retainTileData() {
        if(retainTiles.load(std::memory_order_relaxed))return true;
        if(!blocks.stats().evictions)return false;
        retainTiles.store(true,std::memory_order_relaxed);return true;
    }
    void requested(const std::filesystem::path& path,uint32_t level) {
        if(traceFile.empty())return;
        std::lock_guard<std::mutex> lock(traceMutex);requestedLevels[path.u8string()].insert(level);
    }
    void writeTrace() {
        if(traceFile.empty())return;
        Json result=Json::array();std::lock_guard<std::mutex> lock(traceMutex);
        for(const auto& [path,levels]:requestedLevels)result.push_back({{"path",path},{"levels",levels}});
        std::ofstream output(std::filesystem::u8path(traceFile));output<<result.dump(2);
        if(!output)throw std::runtime_error("Could not save texture request trace");
    }
    void clear() {
        blocks.clear();if(tiles)tiles->clear();usage->peak=usage->bytes.load();
        blocks.setCapacity(totalBudget);workspaceBudget=totalBudget;tileBudget=0;if(tiles)tiles->setCapacity(0);
        retainTiles=false;
        mipLoads=0;sourceBytes=0;sourceReadUs=0;decodeUs=0;
        tileLoads=0;tileBytes=0;tileFallbacks=0;tileErrors=0;decodedBytes=0;deliveredBytes=0;
    }
    template<class Fill>Lease get(const std::string& key,size_t bytes,Fill&& fill) {
        return blocks.get(key,bytes,std::forward<Fill>(fill));
    }
    std::pair<uint64_t,uint64_t> stats() const {
        const auto resident=usage->bytes.load(std::memory_order_relaxed);
        return {resident,std::max(resident,usage->peak.load(std::memory_order_relaxed))};
    }
};
class KtxImage final:public imageSource::ImageSourceBase {
    struct Level {uint64_t offset,bytes,decoded;};
    std::filesystem::path path;
    std::shared_ptr<MipCache> cache;
    imageSource::TextureInfo info{};
    std::vector<Level> levels;
    uint32_t blockBytes=0,scheme=0,imageId=0;
    uint32_t componentMapping=0x03020100u,orientation=0;
    std::unique_ptr<texturecache::Reader> tileCache;
    std::filesystem::path tileRoot;
    texturecache::Source tileInfo{};
    std::once_flag tileOpened;
    std::atomic<bool> tileCacheFailed{false};
    mutable std::mutex statsMutex;
    uint64_t reads=0,bytesRead=0;
    template<class T>static T at(const char* p){T v;std::memcpy(&v,p,sizeof(v));return v;}
    MipCache::Lease mip(uint32_t level) {
        if(level>=levels.size())throw std::runtime_error("KTX mip index out of bounds");
        const auto l=levels[level];
        return cache->get(path.u8string()+":"+std::to_string(level),size_t(l.decoded),[&](char* dst){
            const auto readStart=Clock::now();
            std::ifstream input(path,std::ios::binary);input.seekg(l.offset);
            if(scheme==0)input.read(dst,l.bytes);
            else {
                std::vector<char> encoded(size_t(l.bytes));input.read(encoded.data(),encoded.size());
                if(!input)throw std::runtime_error("KTX mip read failed: "+path.u8string());
                cache->sourceReadUs.fetch_add(uint64_t(elapsedMs(readStart)*1000),std::memory_order_relaxed);
                const auto decodeStart=Clock::now();
                const auto decoded=ZSTD_decompress(dst,size_t(l.decoded),encoded.data(),encoded.size());
                cache->decodeUs.fetch_add(uint64_t(elapsedMs(decodeStart)*1000),std::memory_order_relaxed);
                if(ZSTD_isError(decoded)||decoded!=l.decoded)throw std::runtime_error("KTX Zstd decode failed: "+path.u8string());
            }
            if(!input)throw std::runtime_error("KTX mip read failed: "+path.u8string());
            if(scheme==0)cache->sourceReadUs.fetch_add(uint64_t(elapsedMs(readStart)*1000),std::memory_order_relaxed);
            cache->mipLoads.fetch_add(1,std::memory_order_relaxed);cache->sourceBytes.fetch_add(l.bytes,std::memory_order_relaxed);
            cache->decodedBytes.fetch_add(l.decoded,std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(statsMutex);bytesRead+=l.bytes;
        });
    }
public:
    KtxImage(std::filesystem::path file,std::shared_ptr<MipCache> pool,const std::filesystem::path& tileRoot):path(std::move(file)),cache(std::move(pool)) {
        std::ifstream stream(path,std::ios::binary);char header[80]{};stream.read(header,80);
        const unsigned char magic[]={0xab,0x4b,0x54,0x58,0x20,0x32,0x30,0xbb,0x0d,0x0a,0x1a,0x0a};
        if(!stream||std::memcmp(header,magic,12))throw std::runtime_error("Expected KTX2 texture: "+path.u8string());
        const auto format=at<uint32_t>(header+12);
        info.width=at<uint32_t>(header+20);info.height=at<uint32_t>(header+24);info.numMipLevels=at<uint32_t>(header+40);
        scheme=at<uint32_t>(header+44);
        if(!info.width||!info.height||info.width>65536||info.height>65536||!info.numMipLevels||info.numMipLevels>17||
           at<uint32_t>(header+28)!=0||at<uint32_t>(header+32)>1||at<uint32_t>(header+36)!=1||(scheme!=0&&scheme!=2))
            throw std::runtime_error("Unsupported KTX2 dimensions or supercompression: "+path.u8string());
        // The texture descriptor selects sRGB decoding for color bindings.
        // CUDA BC sRGB array formats themselves are not supported here.
        if(format==139){info.format=CU_AD_FORMAT_BC4_UNORM;info.numChannels=1;blockBytes=8;}
        else if(format==141){info.format=CU_AD_FORMAT_BC5_UNORM;info.numChannels=2;blockBytes=16;}
        else if(format==145||format==146){info.format=CU_AD_FORMAT_BC7_UNORM;info.numChannels=4;blockBytes=16;}
        else throw std::runtime_error("Unsupported KTX2 Vulkan format "+std::to_string(format));
        const auto fileBytes=std::filesystem::file_size(path);
        const uint32_t kvOffset=at<uint32_t>(header+56),kvBytes=at<uint32_t>(header+60);
        if(kvBytes) {
            if(kvBytes>(1u<<20)||kvOffset>fileBytes||kvBytes>fileBytes-kvOffset)throw std::runtime_error("Invalid KTX metadata range");
            std::vector<char> kv(kvBytes);stream.seekg(kvOffset);stream.read(kv.data(),kv.size());
            if(!stream)throw std::runtime_error("KTX metadata read failed");
            for(size_t p=0;p<kv.size();) {
                if(kv.size()-p<4)throw std::runtime_error("Truncated KTX metadata entry");
                const size_t length=at<uint32_t>(kv.data()+p);p+=4;
                if(!length||length>kv.size()-p)throw std::runtime_error("Invalid KTX metadata entry");
                const char* text=kv.data()+p;const auto* nul=static_cast<const char*>(std::memchr(text,0,length));
                if(!nul)throw std::runtime_error("Invalid KTX metadata key");
                const std::string key(text,nul);const size_t valueLength=length-size_t(nul-text)-1;
                if(key=="KTXswizzle") {
                    if(valueLength<4)throw std::runtime_error("Invalid KTX channel swizzle");componentMapping=0;
                    for(uint32_t k=0;k<4;++k){const auto code=std::string("rgba01").find(nul[1+k]);if(code==std::string::npos)throw std::runtime_error("Unknown KTX channel swizzle");componentMapping|=uint32_t(code)<<(k*8);}
                }
                if(key=="KTXorientation") {
                    if(valueLength<2||(nul[1]!='r'&&nul[1]!='l')||(nul[2]!='d'&&nul[2]!='u'))throw std::runtime_error("Unsupported KTX orientation");
                    orientation=(nul[1]=='l'?1u:0u)|(nul[2]=='u'?2u:0u);
                }
                p=(p+length+3)&~size_t(3);
            }
            stream.clear();stream.seekg(80);
        }
        for(uint32_t i=0;i<info.numMipLevels;++i) {
            char bytes[24];stream.read(bytes,24);
            if(!stream)throw std::runtime_error("Truncated KTX level index");
            Level l{at<uint64_t>(bytes),at<uint64_t>(bytes+8),at<uint64_t>(bytes+16)};
            const uint64_t expected=uint64_t((std::max(info.width>>i,1u)+3)/4)*((std::max(info.height>>i,1u)+3)/4)*blockBytes;
            if(l.decoded!=expected||l.offset>fileBytes||l.bytes>fileBytes-l.offset||!l.bytes||(scheme==0&&l.bytes!=l.decoded))
                throw std::runtime_error("Invalid KTX mip range/size: "+path.u8string());
            levels.push_back(l);
        }
        imageId=cache->registerImage(levels.front().decoded);
        info.isValid=true;info.isTiled=true;
        this->tileRoot=tileRoot;tileInfo={format,info.width,info.height,info.numMipLevels,blockBytes};
    }
    void open(imageSource::TextureInfo* out) override{if(out)*out=info;}
    uint32_t swizzle() const{return componentMapping;}
    uint32_t flip() const{return orientation;}
    void close() override{} // Header state is immutable; no retained file handle.
    bool isOpen() const override{return true;}
    const imageSource::TextureInfo& getInfo() const override{return info;}
    CUmemorytype getFillType() const override{return CU_MEMORYTYPE_HOST;}
    bool readTile(char* dest,unsigned int level,const imageSource::Tile& tile,CUstream) override {
        if(tile.width%4||tile.height%4)throw std::runtime_error("BC tile must align to blocks");
        cache->deliveredBytes.fetch_add(size_t(tile.width/4)*(tile.height/4)*blockBytes,std::memory_order_relaxed);
        cache->requested(path,level);
        if(cache->ramTiles) {
            const HostTileKey key{imageId,level,tile.x,tile.y,tile.width,tile.height,blockBytes};
            if(!cache->retainTiles.load(std::memory_order_relaxed)||!cache->tiles->read(key,dest)) {
                auto source=mip(level);
                const BcMipView view{source.data(),source.size(),std::max(info.width>>level,1u),std::max(info.height>>level,1u),blockBytes};
                if(cache->retainTileData()) {
                    cache->reserveTiles(9ull*(tile.width/4)*(tile.height/4)*blockBytes);
                    cache->tiles->fill(key,view,dest);
                } else copyBcTile(key,view,dest);
            }
            std::lock_guard<std::mutex> lock(statsMutex);++reads;return true;
        }
        // Index validation is IO work too. Defer it to the first tile reader
        // instead of opening thousands of cache files before the first image.
        std::call_once(tileOpened,[&]{tileCache=texturecache::Reader::open(tileRoot,path,tileInfo);});
        if(tileCache && !tileCacheFailed && tileCache->contains(level))try {
            const uint32_t side=texturecache::TileSide/4,tw=tile.width/4,th=tile.height/4;
            const uint32_t width=(std::max(info.width>>level,1u)+3)/4,height=(std::max(info.height>>level,1u)+3)/4;
            const uint32_t x=tile.x*tw,y=tile.y*th,xEnd=std::min(x+tw,width),yEnd=std::min(y+th,height);
            std::memset(dest,0,size_t(tw)*th*blockBytes);
            for(uint32_t cy=y/side;cy*side<yEnd;++cy)for(uint32_t cx=x/side;cx*side<xEnd;++cx) {
                const auto key=path.u8string()+":tile:"+std::to_string(level)+":"+std::to_string(cx)+":"+std::to_string(cy);
                auto data=cache->get(key,tileCache->tileBytes(),[&](char* dst){
                    const auto s=tileCache->read(level,cx,cy,dst);cache->tileLoads.fetch_add(1);cache->tileBytes.fetch_add(s.bytes);
                    cache->sourceBytes.fetch_add(s.bytes);cache->sourceReadUs.fetch_add(s.readUs);cache->decodeUs.fetch_add(s.decodeUs);
                    std::lock_guard<std::mutex> lock(statsMutex);bytesRead+=s.bytes;
                });
                const uint32_t left=std::max(x,cx*side),right=std::min(xEnd,(cx+1)*side),top=std::max(y,cy*side),bottom=std::min(yEnd,(cy+1)*side);
                for(uint32_t row=top;row<bottom;++row)
                    std::memcpy(dest+(size_t(row-y)*tw+left-x)*blockBytes,
                                data.data()+(size_t(row-cy*side)*side+left-cx*side)*blockBytes,size_t(right-left)*blockBytes);
            }
            std::lock_guard<std::mutex> lock(statsMutex);++reads;return true;
        } catch(const std::exception& error) {
            tileCacheFailed=true;cache->tileErrors.fetch_add(1);
            std::cerr<<"[texture cache fallback] "<<path.u8string()<<": "<<error.what()<<std::endl;
        }
        cache->tileFallbacks.fetch_add(1);
        auto source=mip(level);
        const uint32_t width=(std::max(info.width>>level,1u)+3)/4,height=(std::max(info.height>>level,1u)+3)/4;
        const uint32_t tw=tile.width/4,th=tile.height/4,x=tile.x*tw,y=tile.y*th;
        std::memset(dest,0,size_t(tw)*th*blockBytes);
        if(x<width&&y<height)for(uint32_t row=0;row<std::min(th,height-y);++row)
            std::memcpy(dest+size_t(row)*tw*blockBytes,source.data()+(size_t(y+row)*width+x)*blockBytes,size_t(std::min(tw,width-x))*blockBytes);
        std::lock_guard<std::mutex> lock(statsMutex);++reads;return true;
    }
    bool readMipLevel(char* dest,unsigned int level,unsigned int width,unsigned int height,CUstream) override {
        if(level>=levels.size()||width!=std::max(info.width>>level,1u)||height!=std::max(info.height>>level,1u))
            throw std::runtime_error("KTX mip dimensions do not match CUDA array");
        auto source=mip(level);std::memcpy(dest,source.data(),size_t(levels[level].decoded));return true;
    }
    bool readMipTail(char* dest,unsigned int first,unsigned int count,const uint2* dims,CUstream stream) override {
        size_t offset=0;
        for(uint32_t i=first;i<count;++i){readMipLevel(dest+offset,i,dims[i].x,dims[i].y,stream);offset+=size_t(levels[i].decoded);}
        return true;
    }
    std::vector<std::vector<char>> readCoarseMips(unsigned int first) {
        if(first>=levels.size())throw std::runtime_error("KTX coarse mip index out of bounds");
        // KTX2 normally stores the small mips together, smallest first. Read
        // that short range once per file instead of reopening it per mip.
        // Keep unusual, noncontiguous layouts bounded with individual reads.
        const auto start=Clock::now();uint64_t begin=levels[first].offset,end=begin,total=0;
        for(size_t i=first;i<levels.size();++i){begin=std::min(begin,levels[i].offset);end=std::max(end,levels[i].offset+levels[i].bytes);total+=levels[i].bytes;}
        std::ifstream input(path,std::ios::binary);
        std::vector<char> range;
        if(end-begin<=total+4096){range.resize(size_t(end-begin));input.seekg(begin);input.read(range.data(),range.size());}
        if(!input)throw std::runtime_error("KTX coarse mip read failed: "+path.u8string());
        std::vector<std::vector<char>> result(levels.size()-first);uint64_t decodeUs=0;
        for(size_t i=first;i<levels.size();++i) {
            const auto l=levels[i];auto& out=result[i-first];out.resize(size_t(l.decoded));
            std::vector<char> encoded;const char* src;
            if(!range.empty())src=range.data()+size_t(l.offset-begin);
            else {encoded.resize(size_t(l.bytes));input.seekg(l.offset);input.read(encoded.data(),encoded.size());src=encoded.data();}
            if(!input)throw std::runtime_error("KTX coarse mip read failed: "+path.u8string());
            const auto decodeStart=Clock::now();
            if(scheme==0)std::memcpy(out.data(),src,out.size());
            else {
                const auto decoded=ZSTD_decompress(out.data(),out.size(),src,size_t(l.bytes));
                if(ZSTD_isError(decoded)||decoded!=l.decoded)throw std::runtime_error("KTX coarse Zstd decode failed: "+path.u8string());
            }
            decodeUs+=uint64_t(elapsedMs(decodeStart)*1000);
        }
        const uint64_t bytes=range.empty()?total:range.size();
        cache->sourceReadUs.fetch_add(uint64_t(elapsedMs(start)*1000)-decodeUs,std::memory_order_relaxed);
        cache->decodeUs.fetch_add(decodeUs,std::memory_order_relaxed);cache->sourceBytes.fetch_add(bytes,std::memory_order_relaxed);
        cache->mipLoads.fetch_add(result.size(),std::memory_order_relaxed);
        {std::lock_guard<std::mutex> lock(statsMutex);bytesRead+=bytes;}
        return result;
    }
    bool readBaseColor(float4&) override{return false;}
    void resetStatistics(){std::lock_guard<std::mutex> lock(statsMutex);reads=0;bytesRead=0;}
    unsigned long long getNumTilesRead() const override{std::lock_guard<std::mutex> lock(statsMutex);return reads;}
    unsigned long long getNumBytesRead() const override{std::lock_guard<std::mutex> lock(statsMutex);return bytesRead;}
};
// Keep the source's 32-texel-and-smaller mips for every texture resident from
// the first frame. Layered BC arrays pack many small textures into a handful
// of allocations instead of spending one 64 KiB sparse tail per texture.
// CUDA performs the same BC decode, sRGB conversion and filtering as the
// streamed texture, including wrap modes and the remaining mip pyramid.
class CoarseTextures {
    struct Item {std::shared_ptr<KtxImage> image;unsigned int id,first;};
    struct Group {
        CUDA_ARRAY3D_DESCRIPTOR arrayDesc{};
        CUDA_TEXTURE_DESC textureDesc{};
        unsigned int levels=0;
        std::vector<Item> items;
        CUmipmappedArray array=nullptr;
        CUtexObject texture=0;
        ~Group(){if(texture)cuTexObjectDestroy(texture);if(array)cuMipmappedArrayDestroy(array);}
    };
    std::vector<std::unique_ptr<Group>> groups;
    std::map<std::array<unsigned int,11>,Group*> active;
    unsigned int maxLayers=0,side=32;
public:
    uint64_t deviceBytes=0,sourceBytes=0;
    CoarseTextures() {
        if(const auto* value=std::getenv("ZENO_CLAS_TEXTURE_COARSE_SIDE")) {
            const auto requested=std::strtoul(value,nullptr,10);
            if(requested>=32 && requested<=256)side=unsigned(requested);
        }
    }
    void add(unsigned int id,std::shared_ptr<KtxImage> image,const demandLoading::TextureDescriptor& desc) {
        if(!maxLayers) {
            CUdevice device;int limit;check(cuCtxGetDevice(&device));
            check(cuDeviceGetAttribute(&limit,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,device));maxLayers=unsigned(limit);
        }
        const auto& info=image->getInfo();unsigned int first=0;
        while(first+1<info.numMipLevels&&std::max(info.width>>first,info.height>>first)>side)++first;
        const auto width=std::max(info.width>>first,1u),height=std::max(info.height>>first,1u),levels=info.numMipLevels-first;
        const std::array<unsigned int,11> key={unsigned(info.format),info.numChannels,width,height,levels,
            unsigned(desc.addressMode[0]),unsigned(desc.addressMode[1]),desc.filterMode,unsigned(desc.mipmapFilterMode),desc.flags,desc.maxAnisotropy};
        auto& group=active[key];
        if(!group||group->items.size()==maxLayers) {
            groups.emplace_back(new Group);group=groups.back().get();
            group->arrayDesc.Width=width;group->arrayDesc.Height=height;group->arrayDesc.Format=info.format;
            group->arrayDesc.NumChannels=info.numChannels;group->arrayDesc.Flags=CUDA_ARRAY3D_LAYERED;
            group->levels=levels;
            auto& td=group->textureDesc;
            td.addressMode[0]=desc.addressMode[0];td.addressMode[1]=desc.addressMode[1];td.filterMode=demandLoading::toCudaFilterMode(desc.filterMode);
            td.flags=CU_TRSF_NORMALIZED_COORDINATES|desc.flags;td.maxAnisotropy=desc.maxAnisotropy;
            td.mipmapFilterMode=desc.mipmapFilterMode;td.maxMipmapLevelClamp=float(levels-1);
        }
        group->items.push_back({std::move(image),id,first});
    }
    std::vector<CoarseTexture> upload(unsigned int count,uint64_t budget) {
        std::vector<CoarseTexture> bindings(count);
        CUdevice device;check(cuCtxGetDevice(&device));
        for(auto& ptr:groups) {
            auto& g=*ptr;g.arrayDesc.Depth=g.items.size();
            // Query the actual dense allocation requirement with a metadata-
            // only deferred array; charge it to the texture budget before any
            // sparse tile pool is allocated. The resident arrays use ordinary
            // CUDA backing, so there is no mapping work during rendering.
            auto probeDesc=g.arrayDesc;probeDesc.Flags|=CUDA_ARRAY3D_DEFERRED_MAPPING;
            CUmipmappedArray probe=nullptr;check(cuMipmappedArrayCreate(&probe,&probeDesc,g.levels));
            CUDA_ARRAY_MEMORY_REQUIREMENTS requirements{};
            const auto queried=cuMipmappedArrayGetMemoryRequirements(&requirements,probe,device);
            cuMipmappedArrayDestroy(probe);check(queried);
            if(deviceBytes+requirements.size+(16ull<<20)>budget)
                throw std::runtime_error("Texture budget cannot hold the permanent coarse mips and a 16 MiB streaming pool");
            check(cuMipmappedArrayCreate(&g.array,&g.arrayDesc,g.levels));deviceBytes+=requirements.size;
            const auto blockBytes=g.arrayDesc.Format==CU_AD_FORMAT_BC4_UNORM?8u:16u;
            std::vector<std::vector<std::vector<char>>> layers;
            for(const auto& item:g.items)layers.push_back(item.image->readCoarseMips(item.first));
            for(unsigned int level=0;level<g.levels;++level) {
                const auto w=std::max(unsigned(g.arrayDesc.Width)>>level,1u),h=std::max(unsigned(g.arrayDesc.Height)>>level,1u);
                const size_t pitch=size_t((w+3)/4)*blockBytes,rows=(h+3)/4,stride=pitch*rows;
                std::vector<char> bytes(stride*g.items.size());
                for(size_t layer=0;layer<g.items.size();++layer) {
                    if(layers[layer][level].size()!=stride)throw std::runtime_error("KTX coarse mip size mismatch");
                    std::memcpy(bytes.data()+layer*stride,layers[layer][level].data(),stride);
                }
                CUarray array;check(cuMipmappedArrayGetLevel(&array,g.array,level));
                CUDA_MEMCPY3D copy{};copy.srcMemoryType=CU_MEMORYTYPE_HOST;copy.srcHost=bytes.data();copy.srcPitch=pitch;copy.srcHeight=rows;
                copy.dstMemoryType=CU_MEMORYTYPE_ARRAY;copy.dstArray=array;copy.WidthInBytes=pitch;copy.Height=rows;copy.Depth=g.items.size();
                check(cuMemcpy3D(&copy));sourceBytes+=bytes.size();
            }
            CUDA_RESOURCE_DESC resource{};resource.resType=CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;resource.res.mipmap.hMipmappedArray=g.array;
            check(cuTexObjectCreate(&g.texture,&resource,&g.textureDesc,nullptr));
            for(unsigned int layer=0;layer<g.items.size();++layer)bindings[g.items[layer].id]={g.texture,layer,g.items[layer].first};
            // The demand loader owns the readers; release our startup records.
            g.items.clear();g.items.shrink_to_fit();
        }
        active.clear();return bindings;
    }
};
}

struct Materials::Impl {
    std::shared_ptr<MipCache> cache;
    std::unique_ptr<demandLoading::DemandLoader,void(*)(demandLoading::DemandLoader*)> loader{nullptr,demandLoading::destroyDemandLoader};
    struct RequestBatch {
        demandLoading::Ticket ticket;
        unsigned int admitted;
        Clock::time_point created=Clock::now();
        RequestBatch(demandLoading::Ticket t,unsigned int capacity):ticket(std::move(t)),admitted(capacity){}
        int numTasksRemaining() const{return ticket.numTasksRemaining();}
        void wait(){ticket.wait();}
    };
    std::vector<RequestBatch> tickets;
    struct Source {std::shared_ptr<KtxImage> image;demandLoading::TextureDescriptor descriptor;};
    std::vector<Source> sources;
    OpaqueSlotPolicy shadowPolicy;
    demandLoading::Options loaderOptions;
    bool stagedUploads=false;
    bool frozen=false;
    bool prioritizePrimary=true;
    bool measureCoverage=false;
    bool predictEnabled=false,retainEnabled=false,hintsReset=true,predictionActive=false;
    TextureMotion motion;
    TextureFeedbackHints hints{};
    Buffer recentReferences,issuedReferences,hintCounts;
    uint32_t* hostHints=nullptr;
    std::array<uint32_t,TextureFeedbackCounters> latestHints{};
    cudaEvent_t hintsReady{};
    bool hintsRecorded=false,hintsSampled=false;
    bool lastMoving=false;
    unsigned int completedFrames=0;
    double frameAverageMs=0;
    MaterialLaunch launch{};
    CoarseTextures coarse;
    Buffer materialBuffer,placementBuffer,idBuffer,missingBuffer,coarseBuffer;
    Buffer textureHistoryBuffer;
    Buffer coverageBuffer,coverageCounts;
    uint32_t primaryPixels=0,primaryMissingPixels=0;
    uint32_t historyPixels=0;
    std::map<unsigned int,Buffer> primaryFeedback;
    std::map<unsigned int,Buffer> lookAheadFeedback;
    std::map<unsigned int,Buffer> demandFeedback;
    Buffer rankIn,rankOut,rankCounts,rankScratch;
    uint64_t* hostUploadPriorities=nullptr;
    cudaEvent_t uploadPrioritiesReady{};
    bool uploadPrioritiesPending=false;
    size_t rankScratchBytes=0;
    unsigned int residencyEpoch=0;
    bool deliveryEnabled=false,minMipEnabled=false;
    bool materialSetsEnabled=false,budgetEnabled=false;
    TextureBudgetPolicy textureBudget;
    std::unique_ptr<TextureMemorySampler> memorySampler;
    uint64_t memorySequence=0,budgetChanges=0;
    uint64_t budgetSampleFree=0,budgetSampleTotal=0,budgetBuildGrowth=0;
    double budgetChangeMs=0,maxBudgetChangeMs=0;
    uint32_t* hostMiss=nullptr;
    cudaEvent_t missReady{};
    cudaStream_t requestStream{};
    double prepareMs=0,requestMs=0;
    unsigned int requestsAllowed=0,pendingRequests=0;
    unsigned int admissionLimit=0,admissionUnknown=0;
    unsigned int interactiveLimit=64;
    bool boundedSubmissionWindow=false;
    Clock::time_point motionUntil{};
    std::unique_ptr<framecapture::Recorder> qualityRecorder;
    demandLoading::Statistics cachedStats{};
    bool latestMiss=true;
    bool prepared=false;
    explicit Impl(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB) {
        const auto startup=Clock::now();
        if(textureMiB<64||hostMiB<32)throw std::runtime_error("CLAS textures require at least 64 MiB GPU and 32 MiB host cache");
        const auto asset=readJson(manifest);
        auto source=std::filesystem::u8path(asset.at("source").get<std::string>());
        if(!source.is_absolute())source=manifest.parent_path()/source;
        const auto gltf=readJson(source);
        // Imported KTX2 plus the bounded host cache is the disk-free default.
        // A separately baked tile cache is an explicit storage tradeoff.
        std::filesystem::path tileRoot;
        if(const auto* p=std::getenv("ZENO_CLAS_TEXTURE_CACHE"))tileRoot=std::string(p)=="0"?std::filesystem::path{}:std::filesystem::u8path(p);
        const auto* hostTiles=std::getenv("ZENO_CLAS_HOST_TILES");
        const bool useRamTiles=!(hostTiles&&std::string(hostTiles)=="0");
        cache=std::make_shared<MipCache>(uint64_t(hostMiB)<<20,useRamTiles&&tileRoot.empty());
        demandLoading::Options options;
        options.maxTexMemPerDevice=uint64_t(textureMiB)<<20;options.maxThreads=4;
        // Bound invalidation batches and the free-tile reserve to 16 MiB.
        // The SDK default reserves 512 MiB in one eviction burst once full.
        options.maxStagedPages=256;
        options.maxTextures=16384;options.numPages=16*1024*1024;options.numPageTableEntries=32768;
        options.maxPinnedMemory=32ull<<20;
        loader.reset(demandLoading::createDemandLoader(options));
        const auto* scheduler=std::getenv("ZENO_CLAS_TEXTURE_SCHEDULER");
        const auto* priority=std::getenv("ZENO_CLAS_TEXTURE_PRIMARY_PRIORITY");
        prioritizePrimary=!(priority&&std::string(priority)=="0");
        const auto* delivery=std::getenv("ZENO_CLAS_TEXTURE_DELIVERY");
        const auto* minMip=std::getenv("ZENO_CLAS_TEXTURE_MIN_MIP");
        deliveryEnabled=!(delivery&&*delivery=='0');minMipEnabled=!(minMip&&*minMip=='0');
        const auto* sets=std::getenv("ZENO_CLAS_TEXTURE_MATERIAL_SETS");
        const auto* budget=std::getenv("ZENO_CLAS_TEXTURE_BUDGET_GOVERNOR");
        materialSetsEnabled=!(sets&&*sets=='0');budgetEnabled=!(budget&&*budget=='0');
        const auto* coverage=std::getenv("ZENO_FRAME_CAPTURE_COVERAGE");
        measureCoverage=framecapture::enabled()&&!(coverage&&std::string(coverage)=="0");
        if(const auto* quality=std::getenv("ZENO_CLAS_TEXTURE_QUALITY_TRACE");quality&&*quality) {
            measureCoverage=true;
            qualityRecorder=std::make_unique<framecapture::Recorder>(std::filesystem::u8path(quality),"texture-quality",
                "frame,primary_pixels,primary_missing,prepared_queued,gpu_uploads,completed_tiles,submitted_tiles,texture_limit_bytes,budget_changes,budget_resize_ms,texture_target_bytes",true);
        }
        const auto* prediction=std::getenv("ZENO_CLAS_TEXTURE_LOOKAHEAD");
        const auto* retention=std::getenv("ZENO_CLAS_TEXTURE_RETENTION");
        predictEnabled=prediction&&*prediction=='1';retainEnabled=retention&&*retention=='1';
        // Interactive binding updates use a bounded window between frames.
        // Offline rendering retains direct, unrestricted demand uploads.
        const bool stageRequested=!(scheduler&&std::string(scheduler)=="0");
        const auto* window=std::getenv("ZENO_CLAS_TEXTURE_SUBMIT_WINDOW_MS");
        const double windowMs=window?std::clamp(std::strtod(window,nullptr),0.,8.):8.;
        boundedSubmissionWindow=stageRequested&&windowMs>0;
        stagedUploads=stageRequested&&!boundedSubmissionWindow;
        if(stagedUploads)configurePreparedTextureUploads(loader.get());
        std::map<std::pair<size_t,bool>,TextureBinding> textureIds;
        std::map<std::filesystem::path,std::shared_ptr<KtxImage>> images;
        auto binding=[&](const Json& info,bool color=false)->TextureBinding {
            TextureBinding b;
            if(info.is_null()||!info.contains("index"))return b;
            const size_t index=info.at("index");
            const auto transform=info.value("extensions",Json::object()).value("KHR_texture_transform",Json::object());
            if(transform.value("texCoord",info.value("texCoord",0))!=0)throw std::runtime_error("CLAS currently supports TEXCOORD_0 material bindings");
            const auto key=std::make_pair(index,color);
            auto it=textureIds.find(key);
            if(it==textureIds.end()) {
                const auto& texture=gltf.at("textures").at(index);
                const auto& image=gltf.at("images").at(texture.at("source").get<size_t>());
                const auto imagePath=(source.parent_path()/std::filesystem::u8path(image.at("uri").get<std::string>())).lexically_normal();
                auto& reader=images[imagePath];
                if(!reader)reader=std::make_shared<KtxImage>(imagePath,cache,tileRoot);
                demandLoading::TextureDescriptor desc;desc.filterMode=FILTER_BILINEAR;
                if(color) {
                    if(reader->getInfo().format!=CU_AD_FORMAT_BC7_UNORM)throw std::runtime_error("Color KTX2 bindings currently require BC7");
                    desc.flags|=CU_TRSF_SRGB;
                }
                const auto samplers=gltf.value("samplers",Json::array());
                const auto sampler=texture.contains("sampler")?samplers.at(texture.at("sampler").get<size_t>()):Json::object();
                auto wrap=[](int mode){return mode==33071?CU_TR_ADDRESS_MODE_CLAMP:mode==33648?CU_TR_ADDRESS_MODE_MIRROR:CU_TR_ADDRESS_MODE_WRAP;};
                desc.addressMode[0]=wrap(sampler.value("wrapS",10497));desc.addressMode[1]=wrap(sampler.value("wrapT",10497));
                if(sampler.value("magFilter",9729)==9728)desc.filterMode=FILTER_POINT;
                desc.maxAnisotropy=8;
                b.id=int(loader->createTexture(reader,desc).getId());b.width=reader->getInfo().width;b.height=reader->getInfo().height;
                if(b.id!=int(sources.size()))throw std::runtime_error("Unexpected demand texture ID ordering");
                sources.push_back({reader,desc});
                b.swizzle=reader->swizzle();b.flip=reader->flip();
                coarse.add(unsigned(b.id),reader,desc);
                it=textureIds.emplace(key,b).first;
            }
            b=it->second;
            const auto offset=transform.value("offset",std::array<float,2>{0,0}),scale=transform.value("scale",std::array<float,2>{1,1});
            const float angle=transform.value("rotation",0.f),c=std::cos(angle),s=std::sin(angle);
            const float matrix[6]={c*scale[0],-s*scale[1],offset[0],s*scale[0],c*scale[1],offset[1]};
            for(int k=0;k<6;++k){if(!std::isfinite(matrix[k]))throw std::runtime_error("Invalid texture transform");b.transform[k]=matrix[k];}
            return b;
        };
        std::vector<Material> materials;
        for(const auto& m:gltf.value("materials",Json::array())) {
            Material out;
            const auto pbr=m.value("pbrMetallicRoughness",Json::object()),extensions=m.value("extensions",Json::object());
            const auto spec=extensions.value("KHR_materials_specular",Json::object()),trans=extensions.value("KHR_materials_transmission",Json::object());
            const auto base=pbr.value("baseColorFactor",std::array<float,4>{1,1,1,1});
            const auto emission=m.value("emissiveFactor",std::array<float,3>{0,0,0}),specColor=spec.value("specularColorFactor",std::array<float,3>{1,1,1});
            std::copy(base.begin(),base.end(),out.baseColor);std::copy(emission.begin(),emission.end(),out.emission);std::copy(specColor.begin(),specColor.end(),out.specularColor);
            out.metallic=pbr.value("metallicFactor",1.f);out.roughness=pbr.value("roughnessFactor",1.f);
            out.specular=spec.value("specularFactor",1.f);out.transmission=trans.value("transmissionFactor",0.f);
            out.ior=extensions.value("KHR_materials_ior",Json::object()).value("ior",1.5f);
            const auto alpha=m.value("alphaMode",std::string("OPAQUE"));
            if(alpha=="MASK")out.flags|=AlphaMask;else if(alpha=="BLEND")out.flags|=zeno_cluster_device::AlphaBlend;else if(alpha!="OPAQUE")throw std::runtime_error("Unknown glTF alpha mode");
            out.alphaCutoff=m.value("alphaCutoff",.5f);if(m.value("doubleSided",false))out.flags|=DoubleSided;
            if(extensions.contains("KHR_materials_unlit"))out.flags|=Unlit;
            out.textures[BaseColor]=binding(pbr.value("baseColorTexture",Json()),true);out.textures[MetallicRoughness]=binding(pbr.value("metallicRoughnessTexture",Json()));
            out.textures[Normal]=binding(m.value("normalTexture",Json()));out.normalScale=m.value("normalTexture",Json::object()).value("scale",1.f);
            out.textures[Emissive]=binding(m.value("emissiveTexture",Json()),true);out.textures[Specular]=binding(spec.value("specularTexture",Json()));
            out.textures[SpecularColor]=binding(spec.value("specularColorTexture",Json()),true);out.textures[Transmission]=binding(trans.value("transmissionTexture",Json()));
            out.textures[Occlusion]=binding(m.value("occlusionTexture",Json()));out.occlusionStrength=m.value("occlusionTexture",Json::object()).value("strength",1.f);
            if(!(out.flags&(AlphaMask|zeno_cluster_device::AlphaBlend|Unlit)) && out.transmission<=0.f) {
                for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
                    const auto& a=out.textures[i];const auto& b=out.textures[j];
                    if(!sameTextureCoordinates(a,b))continue;
                    auto ad=sources[a.id].descriptor,bd=sources[b.id].descriptor;
                    // sRGB conversion and channel swizzle do not change the
                    // footprint; wrap and filtering differences do.
                    ad.flags&=~CU_TRSF_SRGB;bd.flags&=~CU_TRSF_SRGB;
                    if(ad==bd)out.feedbackSet[i]|=1u<<j;
                }
                for(auto& mask:out.feedbackSet)if(!(mask&(mask-1)))mask=0;
            }
            materials.push_back(out);
        }
        const uint32_t fallback=uint32_t(materials.size());Material defaultMaterial;defaultMaterial.metallic=0;defaultMaterial.roughness=1;materials.push_back(defaultMaterial);
        std::vector<PlacementMaterials> placements;std::vector<uint32_t> ids;
        shadowPolicy=OpaqueSlotPolicy(asset.at("meshes").size());
        for(const auto& p:asset.at("nodes")) {
            const auto& node=gltf.at("nodes").at(p.at("sourceNode").get<size_t>());
            const auto& primitives=gltf.at("meshes").at(node.at("mesh").get<size_t>()).at("primitives");
            PlacementMaterials placement{};placement.offset=uint32_t(ids.size());placement.count=uint32_t(primitives.size());
            const auto matrix=p.at("matrix").get<std::array<float,16>>();
            for(int row=0;row<3;++row)for(int col=0;col<4;++col)placement.transform[row*4+col]=matrix[col*4+row];
            placements.push_back(placement);
            std::vector<uint8_t> opaqueSlots;
            for(const auto& prim:primitives) {
                const auto id=prim.value("material",fallback);
                if(id>=materials.size())throw std::runtime_error("Invalid source material ID");ids.push_back(id);
                const auto& material=materials[id];
                opaqueSlots.push_back(opaqueShadowCandidate(bool(material.flags&(AlphaMask|zeno_cluster_device::AlphaBlend)),
                    bool(material.flags&DoubleSided),material.transmission));
            }
            shadowPolicy.add(p.at("mesh").get<uint32_t>(),opaqueSlots);
        }
        materialBuffer.upload(materials);placementBuffer.upload(placements);idBuffer.upload(ids);
        const auto coarseStart=Clock::now();
        const auto coarseMips=coarse.upload(options.maxTextures,options.maxTexMemPerDevice-sizeof(CoarseTexture)*options.maxTextures);
        coarseBuffer.upload(coarseMips);
        coarse.deviceBytes+=coarseMips.size()*sizeof(CoarseTexture);
        // The SDK rounds resize requests up to its arena size. Round down
        // first so permanent mips plus streamed tiles stay inside our budget.
        const auto arenaBytes=otk::TextureTileAllocator::getRecommendedAllocationSize();
        const auto streamBytes=(options.maxTexMemPerDevice-coarse.deviceBytes)/arenaBytes*arenaBytes;
        if(streamBytes<(16ull<<20))throw std::runtime_error("Texture budget leaves less than 16 MiB for streaming after coarse mips");
        loader->setMaxTextureMemory(streamBytes);
        textureBudget={streamBytes,arenaBytes,streamBytes};
        if(budgetEnabled) {
            size_t free=0,total=0;check(cuMemGetInfo(&free,&total));
            loader->setMaxTextureMemory(textureBudget.initial(free,total));
            memorySampler=std::make_unique<TextureMemorySampler>();
        }
        loaderOptions=options;loaderOptions.maxTexMemPerDevice=textureBudget.limit;
        const auto coarseMs=elapsedMs(coarseStart);
        check(cudaMalloc(&missingBuffer.p,(MissingBuckets+1)*sizeof(uint32_t)));
        check(cudaMemset(missingBuffer.p,0,(MissingBuckets+1)*sizeof(uint32_t)));
        try {
            check(cudaMallocHost(reinterpret_cast<void**>(&hostMiss),3*sizeof(uint32_t)));hostMiss[0]=1;hostMiss[1]=hostMiss[2]=0;
            check(cudaEventCreateWithFlags(&missReady,cudaEventDisableTiming));
            // Background tile copies must not implicitly serialize every
            // legacy-stream render/readback. Publication and request readback
            // have explicit fences in prepare/endPass instead.
            check(cudaStreamCreateWithFlags(&requestStream,cudaStreamNonBlocking));
        } catch(...) {
            if(requestStream)cudaStreamDestroy(requestStream);
            if(missReady)cudaEventDestroy(missReady);
            if(hostMiss)cudaFreeHost(hostMiss);
            throw;
        }
        launch.materials=static_cast<const Material*>(materialBuffer.p);launch.placements=static_cast<const PlacementMaterials*>(placementBuffer.p);
        launch.materialIds=static_cast<const uint32_t*>(idBuffer.p);launch.materialCount=uint32_t(materials.size());launch.placementCount=uint32_t(placements.size());
        launch.missing=static_cast<uint32_t*>(missingBuffer.p);
        launch.coarseTextures=static_cast<const CoarseTexture*>(coarseBuffer.p);
        launch.materialSets=materialSetsEnabled?1u:0u;
        const auto* diagnostics=std::getenv("ZENO_CLAS_TEXTURE_DIAGNOSTICS");
        launch.diagnostics=diagnostics&&diagnostics[0]=='1';
        const auto* buckets=std::getenv("ZENO_CLAS_TEXTURE_MISSING_BUCKETS");
        launch.missingMask=launch.diagnostics || (buckets && std::string(buckets)=="1") ? 0u : MissingBuckets-1;
        const auto* reuse=std::getenv("ZENO_CLAS_TEXTURE_REUSE_VERTICES");
        launch.reuseTriangleVertices=!(reuse && std::string(reuse)=="0");
        const auto* opaque=std::getenv("ZENO_CLAS_OPAQUE_SHADOWS");
        launch.opaqueShadowFastPath=!(opaque && std::string(opaque)=="0");
        const auto* dense=std::getenv("ZENO_CLAS_TEXTURE_DENSE_COARSE");
        launch.denseCoarseSampling=!(dense && std::string(dense)=="0");
        if(!launch.opaqueShadowFastPath)shadowPolicy=OpaqueSlotPolicy(asset.at("meshes").size());
        cachedStats=loader->getStatistics();
        std::cout<<"[CLAS materials] materials="<<materials.size()<<" placements="<<placements.size()<<" textures="<<textureIds.size()<<" GPU budget MiB="<<textureMiB<<" host cache budget MiB="<<hostMiB<<std::endl;
        const auto grouped=std::count_if(materials.begin(),materials.end(),[](const auto& m){return m.feedbackSet[0]||m.feedbackSet[1]||m.feedbackSet[2];});
        std::cout<<"[CLAS texture delivery] coverage="<<deliveryEnabled<<" residentMip="<<minMipEnabled
            <<" materialSets="<<materialSetsEnabled<<" compatibleMaterials="<<grouped<<" budgetGovernor="<<budgetEnabled
            <<" arena MiB="<<(textureBudget.quantum>>20)<<" effective streaming MiB="<<(textureBudget.limit>>20)<<std::endl;
        std::cout<<"[CLAS host cache] mode="<<(cache->ramTiles?"ram-tiles":"whole-mips")
            <<" total="<<cache->totalBudget<<" workspace="<<cache->workspaceBudget<<" tiles="<<cache->tileBudget
            <<" workspace floor="<<cache->workspaceFloor<<" activation="<<(cache->ramTiles?"pressure":"off")<<std::endl;
        std::cout<<"[CLAS textures startup] ready="<<textureIds.size()<<" coarse bytes="<<coarse.deviceBytes<<" BC bytes="<<coarse.sourceBytes
            <<" streaming budget="<<streamBytes<<" coarse ms="<<coarseMs<<" material init ms="<<elapsedMs(startup)<<std::endl;
    }
    ~Impl() {
        memorySampler.reset();
        if(loader){
            advancePreparedTextureUploads(loader.get(),false,false);
            for(auto& ticket:tickets)ticket.wait();drainPreparedTextureUploads(loader.get());
            drainTextureUploads(loader.get());cudaStreamSynchronize(requestStream);cudaStreamSynchronize(0);
            const auto* timing=std::getenv("ZENO_OPTIX_RENDER_TIMING");
            if(timing&&timing[0]=='1')try {
                const auto s=loader->getStatistics();const auto host=cache->stats();
                std::cout<<"[CLAS textures final] textures="<<s.numTextures<<" GPU="<<s.deviceMemoryUsed+coarse.deviceBytes<<" coarse="<<coarse.deviceBytes
                    <<" transferred="<<s.bytesTransferredToDevice<<" tiles="<<s.numTilesRead<<" evictions="<<s.numEvictions
                    <<" host mip="<<host.first<<" host peak="<<host.second<<std::endl;
            }catch(const std::exception& error){std::cerr<<"CLAS final texture statistics: "<<error.what()<<std::endl;}
            loader.reset();
            try {cache->writeTrace();}catch(const std::exception& error){std::cerr<<"Texture request trace: "<<error.what()<<std::endl;}
        }
        if(missReady)cudaEventDestroy(missReady);if(hostMiss)cudaFreeHost(hostMiss);
        if(hintsReady)cudaEventDestroy(hintsReady);if(hostHints)cudaFreeHost(hostHints);
        if(uploadPrioritiesReady)cudaEventDestroy(uploadPrioritiesReady);
        if(hostUploadPriorities)cudaFreeHost(hostUploadPriorities);
        if(requestStream)cudaStreamDestroy(requestStream);
    }
};
Materials::Materials(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB):impl(std::make_unique<Impl>(manifest,textureMiB,hostMiB)){}
Materials::~Materials()=default;
bool Materials::primaryDetailPending() const {
    const auto& p=*impl;
    return p.completedFrames<4 || (p.hintCounts.p?(!p.hintsSampled||p.latestHints[9]!=0):p.latestMiss);
}
const OpaqueSlotPolicy& Materials::opaqueShadowPolicy() const{return impl->shadowPolicy;}
void Materials::beginFrame(){
    if(impl->boundedSubmissionWindow&&impl->stagedUploads)pausePreparedTextureUploads(impl->loader.get());
}
void Materials::setView(const TextureView& view,bool interactive) {
    auto& p=*impl;p.predictionActive=false;
    if(!interactive||!p.predictEnabled){p.motion.reset();return;}
    const auto prediction=p.motion.update(view,float(std::clamp(100./std::max(p.frameAverageMs,25.),1.,4.)));
    p.hintsReset|=prediction.reset;
    auto v=[](const auto& a){return float3{a[0],a[1],a[2]};};
    p.launch.lookAheadEye=v(prediction.eye);p.launch.lookAheadRight=v(prediction.right);
    p.launch.lookAheadUp=v(prediction.up);p.launch.lookAheadFront=v(prediction.front);
    p.launch.lookAheadSpread=prediction.spread;p.launch.lookAheadPhase=p.completedFrames&15u;
    p.predictionActive=prediction.active;
}
void Materials::prepare(bool interactive,uint32_t pixels,bool moving,bool geometryBusy) {
    TimelineRange timeline("CLAS textures: publish mappings");
    auto& p=*impl;
    if(p.boundedSubmissionWindow&&interactive!=p.stagedUploads) {
        if(interactive)configurePreparedTextureUploads(p.loader.get());
        else {wait();releasePreparedTextureUploads(p.loader.get());}
        p.stagedUploads=interactive;
    }
    if(interactive && pixels && pixels!=p.historyPixels) {
        if(p.textureHistoryBuffer.p){check(cudaFree(p.textureHistoryBuffer.p));p.textureHistoryBuffer.p=nullptr;}
        check(cudaMalloc(&p.textureHistoryBuffer.p,size_t(pixels)*sizeof(uint32_t)));
        check(cudaMemsetAsync(p.textureHistoryBuffer.p,0,size_t(pixels)*sizeof(uint32_t),0));
        p.historyPixels=pixels;
        p.hintsReset=true;p.motion.reset();p.predictionActive=false;
        if(p.measureCoverage) {
            if(p.coverageBuffer.p){check(cudaFree(p.coverageBuffer.p));p.coverageBuffer.p=nullptr;}
            check(cudaMalloc(&p.coverageBuffer.p,pixels));
            if(!p.coverageCounts.p)check(cudaMalloc(&p.coverageCounts.p,2*sizeof(uint32_t)));
        }
    }
    p.launch.textureHistory=interactive?static_cast<uint32_t*>(p.textureHistoryBuffer.p):nullptr;
    p.launch.primaryCoverage=interactive?static_cast<unsigned char*>(p.coverageBuffer.p):nullptr;
    if(p.frozen||!interactive)p.launch.lookAheadReferences=nullptr;
    if(p.hintsReady&&p.hintsRecorded) {
        const auto ready=cudaEventQuery(p.hintsReady);
        if(ready!=cudaErrorNotReady){check(ready);std::copy_n(p.hostHints,TextureFeedbackCounters,p.latestHints.begin());p.hintsSampled=true;}
    }
    if(p.uploadPrioritiesPending) {
        const auto ready=cudaEventQuery(p.uploadPrioritiesReady);
        if(ready!=cudaErrorNotReady) {
            check(ready);updatePreparedTexturePriorities(p.loader.get(),p.hostUploadPriorities,128);
            p.uploadPrioritiesPending=false;
        }
    }
    // Stop adding sparse map/copy commands before rays and display readback.
    // CPU preparation continues; already submitted copies publish by event.
    if(interactive)pausePreparedTextureUploads(p.loader.get());
    if(p.prepared){if(p.frozen)missed(false);return;}
    p.launch.lookAheadReferences=nullptr;
    const auto start=Clock::now();
    missed(false);
    p.tickets.erase(std::remove_if(p.tickets.begin(),p.tickets.end(),[](const auto& t){return t.numTasksRemaining()==0;}),p.tickets.end());
    if(interactive && p.textureBudget.limit!=p.loaderOptions.maxTexMemPerDevice) {
        const auto uploads=preparedTextureStats(p.loader.get());
        const bool growth=p.textureBudget.limit>p.loaderOptions.maxTexMemPerDevice;
        const bool idle=!moving && !geometryBusy && p.tickets.empty() && !uploads.queued && !uploads.inFlight &&
            !pendingTextureUploads(p.loader.get()) && cudaStreamQuery(p.requestStream)==cudaSuccess && cudaEventQuery(p.missReady)==cudaSuccess;
        if(growth || idle) {
            uint64_t backingBefore=0,backingAfter=0;
            if(!growth) {
                backingBefore=textureTileBackingBytes(p.loader.get());
                // Skip unused ceiling space. Reclaim at most one physically
                // allocated arena, rather than taking minutes to reach it.
                const auto next=backingBefore>p.textureBudget.quantum?backingBefore-p.textureBudget.quantum:0;
                p.textureBudget.limit=std::max(p.textureBudget.floor(),std::min(p.textureBudget.limit,next));
            }
            const auto resizeStart=Clock::now();p.loader->setMaxTextureMemory(p.textureBudget.limit);
            p.loaderOptions.maxTexMemPerDevice=p.textureBudget.limit;++p.budgetChanges;
            p.budgetChangeMs=elapsedMs(resizeStart);p.maxBudgetChangeMs=std::max(p.maxBudgetChangeMs,p.budgetChangeMs);
            if(!growth)backingAfter=textureTileBackingBytes(p.loader.get());
            p.hints.poolPages=uint32_t(p.textureBudget.limit/(64u<<10));
            std::cout<<"[CLAS texture budget] stream MiB="<<(p.textureBudget.limit>>20)<<" free MiB="<<(p.budgetSampleFree>>20)
                <<" reserve MiB="<<(p.textureBudget.reserve(p.budgetSampleTotal,p.budgetBuildGrowth)>>20)
                <<" released MiB="<<((backingBefore-backingAfter)>>20)<<" resize ms="<<p.budgetChangeMs<<std::endl;
        }
    }
    // The build-tree OTK adapter adds mappings only after each worker's
    // upload has completed. Publish those ready mappings/invalidations in
    // render order; incomplete uploads keep their old coarse fallback.
    // Waiting for requestStream here would queue rays behind unrelated
    // sparse map/unmap work, causing long frames once eviction starts.
    if(!p.loader->launchPrepare(0,p.launch.textures))throw std::runtime_error("CUDA device does not support sparse demand textures");
    p.launch.primaryReferences=nullptr;
    p.launch.demandImportance=nullptr;p.launch.minMip=p.minMipEnabled && interactive;
    if(++p.residencyEpoch==0)p.residencyEpoch=1;
    p.launch.residencyEpoch=p.residencyEpoch;
    if(interactive) {
        // Pair feedback storage with the SDK launch context. It cannot be
        // reused until that context's request readback/callback has retired.
        // A shared bitset cleared on stream 0 would race the request stream.
        auto& buffer=p.primaryFeedback[p.launch.textures.poolIndex];
        const auto words=(p.launch.textures.maxNumPages+31u)/32u;
        if(!buffer.p)check(cudaMalloc(&buffer.p,size_t(words+1)*sizeof(uint32_t)));
        p.launch.primaryReferences=static_cast<uint32_t*>(buffer.p);
        if((p.deliveryEnabled || p.materialSetsEnabled) && !p.frozen) {
            auto& demand=p.demandFeedback[p.launch.textures.poolIndex];
            if(!demand.p)check(cudaMalloc(&demand.p,TextureDemandSlots*sizeof(TextureDemand)));
            p.launch.demandImportance=static_cast<TextureDemand*>(demand.p);
            if(!p.rankIn.p) {
                check(cudaMalloc(&p.rankIn.p,TextureDemandSlots*sizeof(unsigned long long)));
                check(cudaMalloc(&p.rankOut.p,TextureDemandSlots*sizeof(unsigned long long)));
                check(cudaMalloc(&p.rankCounts.p,3*sizeof(uint32_t)));
                p.rankScratchBytes=textureImportanceScratchBytes();check(cudaMalloc(&p.rankScratch.p,p.rankScratchBytes));
                const auto* priority=std::getenv("ZENO_CLAS_TEXTURE_UPLOAD_PRIORITY");
                if(!priority||*priority!='0') {
                    check(cudaMallocHost(reinterpret_cast<void**>(&p.hostUploadPriorities),128*sizeof(uint64_t)));
                    check(cudaEventCreateWithFlags(&p.uploadPrioritiesReady,cudaEventDisableTiming));
                }
            }
        }

        if(p.predictEnabled||p.retainEnabled) {
            const auto bytes=size_t(words)*sizeof(uint32_t);
            if(!p.hintCounts.p) {
                check(cudaMalloc(&p.hintCounts.p,TextureFeedbackCounters*sizeof(uint32_t)));
                check(cudaMallocHost(reinterpret_cast<void**>(&p.hostHints),TextureFeedbackCounters*sizeof(uint32_t)));
                std::fill_n(p.hostHints,TextureFeedbackCounters,0u);
                check(cudaEventCreateWithFlags(&p.hintsReady,cudaEventDisableTiming));
                if(p.retainEnabled)check(cudaMalloc(&p.recentReferences.p,4*bytes));
                if(p.predictEnabled)check(cudaMalloc(&p.issuedReferences.p,bytes));
                p.hintsReset=true;
            }
            p.hints.recent=static_cast<uint32_t*>(p.recentReferences.p);p.hints.issued=static_cast<uint32_t*>(p.issuedReferences.p);
            p.hints.counters=static_cast<uint32_t*>(p.hintCounts.p);
            p.hints.firstTilePage=p.loaderOptions.numPageTableEntries;
            p.hints.poolPages=uint32_t(p.loaderOptions.maxTexMemPerDevice/(64u<<10));
            p.hints.retentionPages=p.retainEnabled?std::min(2048u,p.hints.poolPages/16):0u;
            if(p.predictEnabled) {
                auto& ahead=p.lookAheadFeedback[p.launch.textures.poolIndex];
                if(!ahead.p)check(cudaMalloc(&ahead.p,bytes));
                if(!p.frozen&&p.predictionActive&&p.completedFrames%4==1)p.launch.lookAheadReferences=static_cast<uint32_t*>(ahead.p);
            }
        }
    }
    p.prepared=true;
    p.prepareMs=elapsedMs(start);
}
void Materials::beginPass(){
    auto& p=*impl;
    check(cudaMemsetAsync(p.launch.missing,0,(MissingBuckets+1)*sizeof(uint32_t),0));
    if(p.launch.primaryCoverage)check(cudaMemsetAsync(p.launch.primaryCoverage,0,p.historyPixels,0));
    if(p.launch.primaryReferences)
        check(cudaMemsetAsync(p.launch.primaryReferences,0,size_t((p.launch.textures.maxNumPages+31u)/32u)*sizeof(uint32_t),0));
    if(p.launch.lookAheadReferences)
        check(cudaMemsetAsync(p.launch.lookAheadReferences,0,size_t((p.launch.textures.maxNumPages+31u)/32u)*sizeof(uint32_t),0));
    if(p.launch.demandImportance)check(cudaMemsetAsync(p.launch.demandImportance,0,TextureDemandSlots*sizeof(TextureDemand),0));
}
void Materials::endPass(bool interactive,bool moving) {
    TimelineRange timeline("CLAS textures: request readback");
    auto& p=*impl;
    const auto words=p.launch.missingMask+1;
    check(reduceTextureMisses(p.launch.missing,words,0));
    check(cudaMemcpyAsync(p.hostMiss,p.launch.missing+words,sizeof(uint32_t),cudaMemcpyDeviceToHost,0));
    if(p.launch.primaryCoverage) {
        check(reducePrimaryCoverage(p.launch.primaryCoverage,p.historyPixels,static_cast<uint32_t*>(p.coverageCounts.p),0));
        check(cudaMemcpyAsync(p.hostMiss+1,p.coverageCounts.p,2*sizeof(uint32_t),cudaMemcpyDeviceToHost,0));
    }
    check(cudaEventRecord(p.missReady,0));
    // Retain this launch context while frozen. No processRequests means no
    // scans, new workers, mappings or eviction. Resume returns it normally.
    if(p.frozen){p.requestsAllowed=p.pendingRequests=p.admissionLimit=p.admissionUnknown=0;p.requestMs=0;return;}
    const auto start=Clock::now();
    // Requests and LRU references belong to the completed render, including
    // the miss flag readback. Do not scan/recycle that context beforehand.
    check(cudaStreamWaitEvent(p.requestStream,p.missReady,0));
    auto requests=p.launch.textures;
    if(p.boundedSubmissionWindow&&moving)p.motionUntil=Clock::now()+std::chrono::milliseconds(200);
    p.lastMoving=moving||(p.boundedSubmissionWindow&&Clock::now()<p.motionUntil);
    p.admissionLimit=p.admissionUnknown=0;
    if(!interactive)advancePreparedTextureUploads(p.loader.get(),false,false);
    if(interactive) {
        // OTK allocates the full request buffer. Its per-launch capacity can
        // be smaller, bounding live work without throttling offline warm-up.
        // Sparse map/unmap operations also delay subsequent GPU work on WDDM.
        // Bound both admission per frame and total queued tiles; otherwise a
        // busy upload stream can hold the next publication fence for 100+ ms.
        // Offline warm-up retains the SDK's full request capacity.
        // CUDA sparse bindings can delay ray dispatch on WDDM even across
        // nonblocking streams. Admit smaller bursts during camera motion;
        // stationary refinement can recover throughput when there is headroom.
        // Startup compilation is not a texture upload timing signal. A
        // four-page floor driven by total frame time starved cold views, and
        // expensive shading prevented recovery. Keep motion conservative;
        // a stationary view can fill a bounded 4 MiB batch per frame.
        const unsigned int requestsPerFrame=p.stagedUploads?(p.boundedSubmissionWindow?(p.lastMoving?16u:64u):256u):moving?8u:p.interactiveLimit;
        const unsigned int maxPending=p.stagedUploads?(p.boundedSubmissionWindow?128u:768u):2*requestsPerFrame;
        p.admissionLimit=maxPending;
        unsigned int pending=pendingTextureUploads(p.loader.get())+preparedTextureStats(p.loader.get()).queued;
        // Preserve the established conservative quota for unresolved feedback,
        // including partial/empty batches. It must also cover the original
        // admission when the motion limit drops, without loosening backpressure.
        for(const auto& ticket:p.tickets) {
            const auto n=ticket.numTasksRemaining();
            const auto reserve=std::max(ticket.admitted,requestsPerFrame);
            pending+=n<0?reserve:unsigned(n);
            if(n<0)p.admissionUnknown+=reserve;
        }
        requests.requestedPages.capacity=std::min({requests.requestedPages.capacity,requestsPerFrame,pending>=maxPending?0u:maxPending-pending});
        p.pendingRequests=pending;
    }
    p.requestsAllowed=requests.requestedPages.capacity;
    if(p.launch.primaryReferences) {
        const auto words=(p.launch.textures.maxNumPages+31u)/32u;
        if(p.hintCounts.p) {
            // History and counters are owned exclusively by requestStream.
            // No render-stream write races its previous asynchronous scan.
            if(p.hintsReset) {
                if(p.recentReferences.p)check(cudaMemsetAsync(p.recentReferences.p,0,size_t(words)*4*sizeof(uint32_t),p.requestStream));
                if(p.issuedReferences.p)check(cudaMemsetAsync(p.issuedReferences.p,0,size_t(words)*sizeof(uint32_t),p.requestStream));
                check(cudaMemsetAsync(p.hintCounts.p,0,TextureFeedbackCounters*sizeof(uint32_t),p.requestStream));
                p.hintsReset=false;
            }
            p.hints.predicted=p.launch.lookAheadReferences;p.hints.frame=p.completedFrames;
            p.hints.prefetchPages=p.launch.lookAheadReferences?std::min(2u,requests.requestedPages.capacity):0u;
            p.hints.requestPages=requests.requestedPages.capacity;
            check(hintTextureRequests(requests.referenceBits,p.launch.primaryReferences,requests.residenceBits,words,
                p.prioritizePrimary&&p.completedFrames%4!=3,p.hints,p.requestStream));
            check(cudaMemcpyAsync(p.hostHints,p.hintCounts.p,TextureFeedbackCounters*sizeof(uint32_t),cudaMemcpyDeviceToHost,p.requestStream));
            check(cudaEventRecord(p.hintsReady,p.requestStream));
            p.hintsRecorded=true;
        } else check(prioritizeTextureRequests(requests.referenceBits,p.launch.primaryReferences,requests.residenceBits,
            words,p.prioritizePrimary&&p.completedFrames%4!=3,p.requestStream));
        if(p.launch.demandImportance && requests.requestedPages.capacity && p.prioritizePrimary && p.completedFrames%4!=3) {
            check(rankTextureRequests(requests.referenceBits,p.launch.primaryReferences,requests.residenceBits,words,
                p.loaderOptions.numPageTableEntries,requests.requestedPages.capacity,128,p.launch.demandImportance,
                static_cast<unsigned long long*>(p.rankIn.p),static_cast<unsigned long long*>(p.rankOut.p),
                static_cast<uint32_t*>(p.rankCounts.p),p.rankScratch.p,p.rankScratchBytes,p.requestStream));
            if(p.hostUploadPriorities&&!p.uploadPrioritiesPending) {
                check(cudaMemcpyAsync(p.hostUploadPriorities,p.rankOut.p,128*sizeof(uint64_t),cudaMemcpyDeviceToHost,p.requestStream));
                check(cudaEventRecord(p.uploadPrioritiesReady,p.requestStream));p.uploadPrioritiesPending=true;
            }
        }
    }
    p.tickets.emplace_back(p.loader->processRequests(p.requestStream,requests),requests.requestedPages.capacity);
    p.requestMs=elapsedMs(start);
    p.prepared=false;
}
void Materials::frameComplete(double milliseconds,bool geometryBusy,uint64_t buildGrowth) {
    auto& p=*impl;
    if(p.frozen)return;
    // Grant background work after the completed display readback, rather than
    // placing every frame's readback behind a newly admitted upload burst.
    // Ignore first-frame setup/reset costs when estimating steady headroom.
    if(++p.completedFrames>3 && milliseconds<200.)
        p.frameAverageMs=p.frameAverageMs?(.875*p.frameAverageMs+.125*milliseconds):milliseconds;
    const double headroom=p.frameAverageMs?std::clamp(36.-p.frameAverageMs,.5,2.):2.;
    if(p.memorySampler) {
        p.budgetBuildGrowth=buildGrowth;
        auto memory=p.memorySampler->sample();
        if(memory.sequence!=p.memorySequence) {
            p.memorySequence=memory.sequence;
            p.budgetSampleFree=memory.free;p.budgetSampleTotal=memory.total;
            // Decide after this frame; apply before the next launch context is
            // prepared, once old feedback has retired. Never resize underneath
            // the request callback submitted by this same frame.
            if(p.textureBudget.limit==p.loaderOptions.maxTexMemPerDevice)
                p.textureBudget.update(memory.free,memory.total,buildGrowth,!p.lastMoving&&!geometryBusy);
            else if(memory.free>p.textureBudget.reserve(memory.total,buildGrowth)+(uint64_t(256)<<20))
                p.textureBudget.limit=p.loaderOptions.maxTexMemPerDevice;
        }
    }
    advancePreparedTextureUploads(p.loader.get(),true,p.lastMoving,headroom,geometryBusy,primaryDetailPending());
    if(p.qualityRecorder) {
        // prepare() already sampled completed feedback. Coverage can lag one
        // frame; this adds no synchronous readback and no extra GPU queries.
        const auto prepared=preparedTextureStats(p.loader.get());const auto completed=completedTextureStats(p.loader.get());
        p.qualityRecorder->record(framecapture::nowMs(),0.,{double(p.completedFrames),double(p.primaryPixels),double(p.primaryMissingPixels),
            double(prepared.queued),double(submittedTextureUploads(p.loader.get())),double(completed.tiles),double(prepared.submitted),
            double(p.loaderOptions.maxTexMemPerDevice),double(p.budgetChanges),p.budgetChangeMs,double(p.textureBudget.limit)});
    }
}
bool Materials::missed(bool wait){
    if(wait)check(cudaEventSynchronize(impl->missReady));
    else {const auto status=cudaEventQuery(impl->missReady);if(status==cudaErrorNotReady)return impl->latestMiss;check(status);}
    if(*impl->hostMiss&2u)throw std::runtime_error("Invalid CLAS material or shading attribute binding");
    impl->primaryPixels=impl->hostMiss[1];impl->primaryMissingPixels=impl->hostMiss[2];
    impl->latestMiss=*impl->hostMiss!=0;return impl->latestMiss;
}
void Materials::wait(){
    advancePreparedTextureUploads(impl->loader.get(),false,false);
    for(auto& t:impl->tickets)t.wait();drainPreparedTextureUploads(impl->loader.get());
    drainTextureUploads(impl->loader.get());check(cudaStreamSynchronize(impl->requestStream));impl->tickets.clear();
}
void Materials::setFrozen(bool frozen) {
    auto& p=*impl;if(frozen==p.frozen)return;
    if(frozen){wait();check(cudaStreamSynchronize(0));}
    else if(p.prepared) {
        // A held SDK context accumulated references while frozen. Start the
        // resumed frame with its current view, not requests from that interval.
        check(cudaMemsetAsync(p.launch.textures.referenceBits,0,
            size_t((p.launch.textures.maxNumPages+31u)/32u)*sizeof(uint32_t),0));
    }
    p.frozen=frozen;
}
void Materials::clearResidency(){
    const auto start=Clock::now();auto& p=*impl;
    wait();check(cudaStreamSynchronize(0));
    p.uploadPrioritiesPending=false;
    const auto retired=p.loader->getStatistics();
    p.loader.reset();p.primaryFeedback.clear();p.lookAheadFeedback.clear();p.demandFeedback.clear();p.launch.primaryReferences=nullptr;
    p.launch.lookAheadReferences=nullptr;p.hintsReset=true;p.hintsRecorded=p.hintsSampled=false;p.motion.reset();p.latestHints.fill(0);p.cache->clear();
    for(const auto& source:p.sources)source.image->resetStatistics();
    p.loader.reset(demandLoading::createDemandLoader(p.loaderOptions));
    if(p.stagedUploads)configurePreparedTextureUploads(p.loader.get());
    updatePreparedTexturePriorities(p.loader.get(),nullptr,0);
    for(size_t i=0;i<p.sources.size();++i)
        if(p.loader->createTexture(p.sources[i].image,p.sources[i].descriptor).getId()!=i)
            throw std::runtime_error("Texture residency reset changed material bindings");
    p.tickets.clear();p.launch.textures={};p.prepared=false;p.latestMiss=true;*p.hostMiss=1;
    p.interactiveLimit=64;p.completedFrames=0;p.frameAverageMs=0;p.motionUntil={};p.cachedStats=p.loader->getStatistics();
    std::cout<<"[CLAS textures reset] streamed GPU tiles and host data cleared; coarse mips retained; retired evictions="<<retired.numEvictions<<" transferred="<<retired.bytesTransferredToDevice<<" bytes; ms="<<elapsedMs(start)<<std::endl;
}
bool Materials::settled() const{return !impl->latestMiss&&pendingTextureUploads(impl->loader.get())==0&&preparedTextureStats(impl->loader.get()).queued==0&&std::all_of(impl->tickets.begin(),impl->tickets.end(),[](const auto& t){return t.numTasksRemaining()==0;});}
MaterialQueueStats Materials::queueStats() const {
    const auto& p=*impl;MaterialQueueStats s;
    s.ticketSlots=p.tickets.size();
    for(const auto& batch:p.tickets) {
        const auto remaining=batch.numTasksRemaining();if(!remaining){++s.completedTickets;continue;}
        ++s.batches;if(remaining<0)++s.unknownBatches;else s.cpuTasks+=unsigned(remaining);
        s.oldestBatchMs=std::max(s.oldestBatchMs,elapsedMs(batch.created));
    }
    s.uploads=pendingTextureUploads(p.loader.get());s.prepared=preparedTextureStats(p.loader.get()).queued;
    s.feedbackContexts=p.primaryFeedback.size();
    s.admissionLimit=p.admissionLimit;s.admissionPending=p.pendingRequests;
    s.admissionUnknown=p.admissionUnknown;s.admissionAllowed=p.requestsAllowed;
    const auto callbacks=textureCallbackStats();
    s.callbacksQueued=callbacks.queued;s.callbacksActive=callbacks.active;s.callbacksPeak=callbacks.peak;
    s.callbacksCompleted=callbacks.completed;s.oldestCallbackMs=callbacks.oldestMs;s.activeCallbackMs=callbacks.activeMs;
    return s;
}
zeno_cluster_device::MaterialLaunch Materials::launch(bool twoSidedOpaque,bool opaqueShadowAllowed) const {
    auto result=impl->launch;result.twoSidedOpaque=twoSidedOpaque?1u:0u;
    result.opaqueShadowFastPath=result.opaqueShadowFastPath && opaqueShadowAllowed;return result;
}
MaterialStats Materials::stats(bool wait) const {
    const auto start=Clock::now();
    auto& p=*impl;
    if(wait)p.cachedStats=p.loader->getStatistics();
    const auto s=p.cachedStats;const auto host=p.cache->stats();const auto hostBlocks=p.cache->blocks.stats();const auto uploads=preparedTextureStats(p.loader.get());const auto completed=completedTextureStats(p.loader.get());
    return {s.numTextures,s.virtualTextureBytes,s.deviceMemoryUsed+p.coarse.deviceBytes,s.bytesTransferredToDevice,s.numTilesRead,s.numEvictions,host.first,host.second,
        impl->cache->mipLoads.load(),impl->cache->sourceBytes.load(),impl->prepareMs,impl->requestMs,elapsedMs(start),
        double(impl->cache->sourceReadUs.load())/1000,double(impl->cache->decodeUs.load())/1000,impl->requestsAllowed,impl->pendingRequests,
        p.cache->tileLoads.load(),p.cache->tileBytes.load(),p.cache->tileFallbacks.load(),p.cache->tileErrors.load(),
        uploads.queued,uploads.inFlight,uploads.prepared,uploads.submitted,uploads.bytes,uploads.queuedBytes,uploads.submitMs,uploads.maxSubmitMs,uploads.readyLatencyMs,uploads.gpuMs,uploads.maxGpuMs,completed.tiles,completed.samplers,
        p.cache->decodedBytes.load(),p.cache->deliveredBytes.load(),hostBlocks.hits,hostBlocks.misses,hostBlocks.reloads,p.primaryPixels,p.primaryMissingPixels,
        completed.eventsCreated,completed.eventsReused,completed.cachedEvents,p.latestHints[1],p.latestHints[6],p.latestHints[4],p.latestHints[5],p.latestHints[0],
        p.latestHints[8],p.latestHints[9],p.latestHints[10],p.latestHints[11],completed.evictions,completed.evictionMs,completed.maxEvictionMs,
        p.cache->tiles?p.cache->tiles->stats():HostTileStats{},hostBlocks,p.cache->workspaceBudget.load(),p.cache->tileBudget.load(),p.cache->ramTiles,p.cache->retainTiles.load()};
}
}
