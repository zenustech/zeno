#include "DecodedGeometryCache.h"
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace zeno::cluster {
namespace {
using Clock=std::chrono::steady_clock;
uint64_t nanos(Clock::time_point start) {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());
}
void validate(const Page& page,const Group& group,bool attributes) {
    if(page.clusters.size()!=group.refined.size() || page.vertices.size()!=group.vertices ||
       page.indices.size()!=uint64_t(group.triangles)*3 ||
       page.attributes.size()!=(attributes?group.vertices:0) ||
       page.materials.size()!=(attributes?group.triangles:0))
        throw std::runtime_error("Decoded geometry page does not match its metadata");
}
template<class T>void pack(char*& out,const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    if(!v.empty())std::memcpy(out,v.data(),v.size()*sizeof(T));
    out+=v.size()*sizeof(T);
}
template<class T>void unpack(const char*& in,std::vector<T>& v,size_t count) {
    v.resize(count);
    if(count)std::memcpy(v.data(),in,count*sizeof(T));
    in+=count*sizeof(T);
}
}
DecodedGeometryCache::DecodedGeometryCache(uint64_t bytes,size_t keys):capacity(bytes),blocks(bytes,usage,false,true),seenKeys((keys+63)/64),keyCount(keys){}
uint64_t DecodedGeometryCache::pageBytes(const Group& g,bool attributes) {
    return uint64_t(g.refined.size())*sizeof(Cluster)+uint64_t(g.vertices)*sizeof(Vertex)+
        uint64_t(g.triangles)*3+(attributes?uint64_t(g.vertices)*sizeof(zeno_cluster_device::ShadingVertex)+uint64_t(g.triangles)*2:0);
}
bool DecodedGeometryCache::read(uint32_t key,const Group& group,bool attributes,Page& output,
                              const std::function<void(Page&)>& decode,bool allowFill) {
    const auto bytes=pageBytes(group,attributes);
    if(!capacity || !bytes || bytes>capacity) {++bypasses;decode(output);return false;}
    if(bytes>std::numeric_limits<size_t>::max())throw std::runtime_error("Decoded geometry page exceeds host addressing");
    // Bucket only larger production pools; charge padding to the RAM budget.
    // Equal-sized evictions can then recycle their allocation without heap churn.
    const auto storage=capacity>=(1u<<20)?std::min(capacity,(bytes+4095)&~uint64_t(4095)):bytes;
    if(keyCount) {
        bool seen;
        {
            std::lock_guard<std::mutex> lock(reuseMutex);
            if(key>=keyCount)throw std::runtime_error("Geometry cache key outside asset");
            const auto bit=uint64_t(1)<<(key&63);
            seen=(seenKeys[key>>6]&bit)!=0;seenKeys[key>>6]|=bit;
        }
        if(!seen){++bypasses;decode(output);return false;}
    }
    const auto name=std::to_string(key);
    bool decoded=false;
    auto fill=[&](char* destination) {
        decode(output);validate(output,group,attributes);decoded=true;
        const auto start=Clock::now();auto* at=destination;
        pack(at,output.clusters);pack(at,output.vertices);pack(at,output.indices);
        pack(at,output.attributes);pack(at,output.materials);
        storedBytes.fetch_add(bytes,std::memory_order_relaxed);storeNs.fetch_add(nanos(start),std::memory_order_relaxed);
    };
    auto cached=allowFill?std::optional<HostBlockCache::Lease>(blocks.get(name,size_t(storage),fill)):
                          blocks.tryGet(name,size_t(storage));
    if(!cached){++bypasses;decode(output);return false;}
    if(decoded)return false;
    const auto start=Clock::now();auto* at=cached->data();
    unpack(at,output.clusters,group.refined.size());unpack(at,output.vertices,group.vertices);
    unpack(at,output.indices,size_t(group.triangles)*3);
    unpack(at,output.attributes,attributes?group.vertices:0);
    unpack(at,output.materials,attributes?group.triangles:0);
    restoredBytes.fetch_add(bytes,std::memory_order_relaxed);restoreNs.fetch_add(nanos(start),std::memory_order_relaxed);
    return true;
}
DecodedGeometryStats DecodedGeometryCache::stats() const {
    return {blocks.stats(),bypasses.load(),restoredBytes.load(),storedBytes.load(),
            double(restoreNs.load())/1.e6,double(storeNs.load())/1.e6};
}
}
