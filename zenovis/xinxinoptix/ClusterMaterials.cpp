#include "ClusterMaterials.h"
#include "CompletedTextureUpload.h"
#include "ClusterProfiling.h"
#include "ClusterCache.h"
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
#include <mutex>
#include <stdexcept>

namespace zeno::cluster {
namespace {
using namespace zeno_cluster_device;
using Json=nlohmann::json;
void check(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
void check(CUresult r){if(r!=CUDA_SUCCESS){const char* message=nullptr;cuGetErrorString(r,&message);throw std::runtime_error(message?message:"CUDA driver error");}}
using Clock=std::chrono::steady_clock;
double elapsedMs(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
struct Buffer {
    void* p=nullptr;
    ~Buffer(){if(p)cudaFree(p);}
    template<class T>void upload(const std::vector<T>& v){check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
// A shared, bounded cache of decompressed BC mip levels. A KTX2 Zstd frame
// covers one whole mip, so GPU tile requests share these host-side frames.
class MipCache {
    struct Entry {std::vector<char> data;uint64_t tick=0;bool loading=true;};
    std::mutex mutex;
    std::condition_variable ready;
    std::map<std::string,std::shared_ptr<Entry>> entries;
    uint64_t budget,tick=0;
    std::atomic<uint64_t> used{0},peak{0};
public:
    std::atomic<uint64_t> mipLoads{0},sourceBytes{0};
    std::atomic<uint64_t> sourceReadUs{0},decodeUs{0};
    struct Lease {
        std::shared_ptr<Entry> entry;
        MipCache* owner;
        Lease(std::shared_ptr<Entry> e,MipCache* o):entry(std::move(e)),owner(o){}
        Lease(const Lease&)=delete;
        ~Lease(){
            // Availability is the wait predicate in get(). Change it under
            // the same mutex so a release cannot notify just before a waiter
            // atomically unlocks and sleeps.
            {std::lock_guard<std::mutex> lock(owner->mutex);entry.reset();}
            owner->ready.notify_all();
        }
        const char* data() const{return entry->data.data();}
    };
    explicit MipCache(uint64_t bytes):budget(bytes){}
    template<class Fill>Lease get(const std::string& key,size_t bytes,Fill fill) {
        std::unique_lock<std::mutex> lock(mutex);
        if(bytes>budget)throw std::runtime_error("Texture mip exceeds the host mip-cache budget");
        for(;;) {
            auto it=entries.find(key);
            if(it!=entries.end()) {
                if(it->second->loading){ready.wait(lock);continue;}
                it->second->tick=++tick;return Lease(it->second,this);
            }
            if(used+bytes>budget) {
                auto oldest=entries.end();
                for(auto i=entries.begin();i!=entries.end();++i)
                    if(!i->second->loading && i->second.use_count()==1 &&
                       (oldest==entries.end()||i->second->tick<oldest->second->tick))oldest=i;
                if(oldest==entries.end()){ready.wait(lock);continue;}
                used-=oldest->second->data.size();entries.erase(oldest);continue;
            }
            auto entry=std::make_shared<Entry>();entry->data.resize(bytes);entry->tick=++tick;
            entries.emplace(key,entry);used+=bytes;peak.store(std::max(peak.load(),used.load()));lock.unlock();
            try {fill(entry->data.data());}
            catch(...) {lock.lock();entries.erase(key);used-=bytes;ready.notify_all();throw;}
            lock.lock();entry->loading=false;ready.notify_all();return Lease(std::move(entry),this);
        }
    }
    std::pair<uint64_t,uint64_t> stats(){const auto resident=used.load(std::memory_order_relaxed);return {resident,std::max(resident,peak.load(std::memory_order_relaxed))};}
};
class KtxImage final:public imageSource::ImageSourceBase {
    struct Level {uint64_t offset,bytes,decoded;};
    std::filesystem::path path;
    std::shared_ptr<MipCache> cache;
    imageSource::TextureInfo info{};
    std::vector<Level> levels;
    uint32_t blockBytes=0,scheme=0;
    uint32_t componentMapping=0x03020100u,orientation=0;
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
            std::lock_guard<std::mutex> lock(statsMutex);bytesRead+=l.bytes;
        });
    }
public:
    KtxImage(std::filesystem::path file,std::shared_ptr<MipCache> pool):path(std::move(file)),cache(std::move(pool)) {
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
        info.isValid=true;info.isTiled=true;
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
    unsigned int maxLayers=0;
public:
    uint64_t deviceBytes=0,sourceBytes=0;
    void add(unsigned int id,std::shared_ptr<KtxImage> image,const demandLoading::TextureDescriptor& desc) {
        if(!maxLayers) {
            CUdevice device;int limit;check(cuCtxGetDevice(&device));
            check(cuDeviceGetAttribute(&limit,CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,device));maxLayers=unsigned(limit);
        }
        const auto& info=image->getInfo();unsigned int first=0;
        while(first+1<info.numMipLevels&&std::max(info.width>>first,info.height>>first)>32)++first;
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
            for(unsigned int layer=0;layer<g.items.size();++layer)bindings[g.items[layer].id]={g.texture,layer,0};
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
    std::vector<demandLoading::Ticket> tickets;
    MaterialLaunch launch{};
    CoarseTextures coarse;
    Buffer materialBuffer,placementBuffer,idBuffer,missingBuffer,coarseBuffer;
    uint32_t* hostMiss=nullptr;
    cudaEvent_t missReady{};
    cudaStream_t requestStream{};
    double prepareMs=0,requestMs=0;
    unsigned int requestsAllowed=0,pendingRequests=0;
    unsigned int interactiveLimit=8,fastFrames=0;
    demandLoading::Statistics cachedStats{};
    bool latestMiss=true;
    bool prepared=false;
    explicit Impl(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB) {
        const auto startup=Clock::now();
        if(textureMiB<64||hostMiB<32)throw std::runtime_error("CLAS textures require at least 64 MiB GPU and 32 MiB host cache");
        cache=std::make_shared<MipCache>(uint64_t(hostMiB)<<20);
        const auto asset=readJson(manifest);
        auto source=std::filesystem::u8path(asset.at("source").get<std::string>());
        if(!source.is_absolute())source=manifest.parent_path()/source;
        const auto gltf=readJson(source);
        demandLoading::Options options;
        options.maxTexMemPerDevice=uint64_t(textureMiB)<<20;options.maxThreads=4;
        // Bound invalidation batches and the free-tile reserve to 16 MiB.
        // The SDK default reserves 512 MiB in one eviction burst once full.
        options.maxStagedPages=256;
        options.maxTextures=16384;options.numPages=16*1024*1024;options.numPageTableEntries=32768;
        options.maxPinnedMemory=32ull<<20;
        loader.reset(demandLoading::createDemandLoader(options));
        std::map<std::pair<size_t,bool>,TextureBinding> textureIds;
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
                auto reader=std::make_shared<KtxImage>(source.parent_path()/std::filesystem::u8path(image.at("uri").get<std::string>()),cache);
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
            materials.push_back(out);
        }
        const uint32_t fallback=uint32_t(materials.size());Material defaultMaterial;defaultMaterial.metallic=0;defaultMaterial.roughness=1;materials.push_back(defaultMaterial);
        std::vector<PlacementMaterials> placements;std::vector<uint32_t> ids;
        for(const auto& p:asset.at("nodes")) {
            const auto& node=gltf.at("nodes").at(p.at("sourceNode").get<size_t>());
            const auto& primitives=gltf.at("meshes").at(node.at("mesh").get<size_t>()).at("primitives");
            PlacementMaterials placement{};placement.offset=uint32_t(ids.size());placement.count=uint32_t(primitives.size());
            const auto matrix=p.at("matrix").get<std::array<float,16>>();
            for(int row=0;row<3;++row)for(int col=0;col<4;++col)placement.transform[row*4+col]=matrix[col*4+row];
            placements.push_back(placement);
            for(const auto& prim:primitives) {
                const auto id=prim.value("material",fallback);
                if(id>=materials.size())throw std::runtime_error("Invalid source material ID");ids.push_back(id);
            }
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
        const auto coarseMs=elapsedMs(coarseStart);
        check(cudaMalloc(&missingBuffer.p,sizeof(uint32_t)));check(cudaMemset(missingBuffer.p,0,sizeof(uint32_t)));
        try {
            check(cudaMallocHost(reinterpret_cast<void**>(&hostMiss),sizeof(uint32_t)));*hostMiss=1;
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
        const auto* diagnostics=std::getenv("ZENO_CLAS_TEXTURE_DIAGNOSTICS");
        launch.diagnostics=diagnostics&&diagnostics[0]=='1';
        cachedStats=loader->getStatistics();
        std::cout<<"[CLAS materials] materials="<<materials.size()<<" placements="<<placements.size()<<" textures="<<textureIds.size()<<" GPU budget MiB="<<textureMiB<<" host mip budget MiB="<<hostMiB<<std::endl;
        std::cout<<"[CLAS textures startup] ready="<<textureIds.size()<<" coarse bytes="<<coarse.deviceBytes<<" BC bytes="<<coarse.sourceBytes
            <<" streaming budget="<<streamBytes<<" coarse ms="<<coarseMs<<" material init ms="<<elapsedMs(startup)<<std::endl;
    }
    ~Impl() {
        if(loader){
            for(auto& ticket:tickets)ticket.wait();cudaStreamSynchronize(requestStream);cudaStreamSynchronize(0);
            const auto* timing=std::getenv("ZENO_OPTIX_RENDER_TIMING");
            if(timing&&timing[0]=='1')try {
                const auto s=loader->getStatistics();const auto host=cache->stats();
                std::cout<<"[CLAS textures final] textures="<<s.numTextures<<" GPU="<<s.deviceMemoryUsed+coarse.deviceBytes<<" coarse="<<coarse.deviceBytes
                    <<" transferred="<<s.bytesTransferredToDevice<<" tiles="<<s.numTilesRead<<" evictions="<<s.numEvictions
                    <<" host mip="<<host.first<<" host peak="<<host.second<<std::endl;
            }catch(const std::exception& error){std::cerr<<"CLAS final texture statistics: "<<error.what()<<std::endl;}
            loader.reset();
        }
        if(missReady)cudaEventDestroy(missReady);if(hostMiss)cudaFreeHost(hostMiss);
        if(requestStream)cudaStreamDestroy(requestStream);
    }
};
Materials::Materials(const std::filesystem::path& manifest,uint32_t textureMiB,uint32_t hostMiB):impl(std::make_unique<Impl>(manifest,textureMiB,hostMiB)){}
Materials::~Materials()=default;
void Materials::prepare() {
    TimelineRange timeline("CLAS textures: publish mappings");
    auto& p=*impl;
    if(p.prepared)return;
    const auto start=Clock::now();
    missed(false);
    p.tickets.erase(std::remove_if(p.tickets.begin(),p.tickets.end(),[](const auto& t){return t.numTasksRemaining()==0;}),p.tickets.end());
    // The build-tree OTK adapter adds mappings only after each worker's
    // upload has completed. Publish those ready mappings/invalidations in
    // render order; incomplete uploads keep their old coarse fallback.
    // Waiting for requestStream here would queue rays behind unrelated
    // sparse map/unmap work, causing long frames once eviction starts.
    if(!p.loader->launchPrepare(0,p.launch.textures))throw std::runtime_error("CUDA device does not support sparse demand textures");
    p.prepared=true;
    p.prepareMs=elapsedMs(start);
}
void Materials::beginPass(){check(cudaMemsetAsync(impl->launch.missing,0,sizeof(uint32_t),0));}
void Materials::endPass(bool interactive,bool moving) {
    TimelineRange timeline("CLAS textures: request readback");
    auto& p=*impl;check(cudaMemcpyAsync(p.hostMiss,p.launch.missing,sizeof(uint32_t),cudaMemcpyDeviceToHost,0));check(cudaEventRecord(p.missReady,0));
    const auto start=Clock::now();
    // Requests and LRU references belong to the completed render, including
    // the miss flag readback. Do not scan/recycle that context beforehand.
    check(cudaStreamWaitEvent(p.requestStream,p.missReady,0));
    auto requests=p.launch.textures;
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
        const unsigned int requestsPerFrame=moving?std::min(8u,p.interactiveLimit):p.interactiveLimit;
        const unsigned int maxPending=2*requestsPerFrame;
        unsigned int pending=pendingTextureUploads(p.loader.get());
        for(const auto& ticket:p.tickets){const auto n=ticket.numTasksRemaining();pending+=n<0?requestsPerFrame:unsigned(n);}
        requests.requestedPages.capacity=std::min({requests.requestedPages.capacity,requestsPerFrame,pending>=maxPending?0u:maxPending-pending});
        p.pendingRequests=pending;
    }
    p.requestsAllowed=requests.requestedPages.capacity;
    p.tickets.push_back(p.loader->processRequests(p.requestStream,requests));
    p.requestMs=elapsedMs(start);
    p.prepared=false;
}
void Materials::frameComplete(double milliseconds) {
    auto& p=*impl;
    if(milliseconds>55.) {
        p.interactiveLimit=std::max(4u,p.interactiveLimit/2);p.fastFrames=0;
    } else if(milliseconds<40.) {
        if(++p.fastFrames>=16){p.interactiveLimit=std::min(16u,p.interactiveLimit+1);p.fastFrames=0;}
    } else p.fastFrames=0;
}
bool Materials::missed(bool wait){
    if(wait)check(cudaEventSynchronize(impl->missReady));
    else {const auto status=cudaEventQuery(impl->missReady);if(status==cudaErrorNotReady)return impl->latestMiss;check(status);}
    if(*impl->hostMiss&2u)throw std::runtime_error("Invalid CLAS material or shading attribute binding");
    impl->latestMiss=*impl->hostMiss!=0;return impl->latestMiss;
}
void Materials::wait(){for(auto& t:impl->tickets)t.wait();drainTextureUploads(impl->loader.get());check(cudaStreamSynchronize(impl->requestStream));impl->tickets.clear();}
bool Materials::settled() const{return !impl->latestMiss&&pendingTextureUploads(impl->loader.get())==0&&std::all_of(impl->tickets.begin(),impl->tickets.end(),[](const auto& t){return t.numTasksRemaining()==0;});}
zeno_cluster_device::MaterialLaunch Materials::launch(bool twoSidedOpaque) const {
    auto result=impl->launch;result.twoSidedOpaque=twoSidedOpaque?1u:0u;return result;
}
MaterialStats Materials::stats(bool wait) const {
    const auto start=Clock::now();
    auto& p=*impl;
    if(wait)p.cachedStats=p.loader->getStatistics();
    const auto s=p.cachedStats;const auto host=p.cache->stats();
    return {s.numTextures,s.virtualTextureBytes,s.deviceMemoryUsed+p.coarse.deviceBytes,s.bytesTransferredToDevice,s.numTilesRead,s.numEvictions,host.first,host.second,
        impl->cache->mipLoads.load(),impl->cache->sourceBytes.load(),impl->prepareMs,impl->requestMs,elapsedMs(start),
        double(impl->cache->sourceReadUs.load())/1000,double(impl->cache->decodeUs.load())/1000,impl->requestsAllowed,impl->pendingRequests};
}
}
