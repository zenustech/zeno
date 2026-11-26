#include "WaterScene.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>

namespace zeno::water {
namespace {
void check(cudaError_t error){if(error)throw std::runtime_error(std::string("Water scene CUDA: ")+cudaGetErrorString(error));}
bool equal(const zeno_water_device::Launch& a,const zeno_water_device::Launch& b) {
    return a.surfaces==b.surfaces && a.patches==b.patches && a.surfaceCount==b.surfaceCount && a.patchCount==b.patchCount;
}
}
struct SceneRuntime::Impl {
    struct Source {std::unique_ptr<Runtime> runtime;unsigned slot=0;bool visible=false;uint64_t nativeVersion=0,version=0;};
    std::map<std::string,Source> sources;
    std::vector<PublishedSurface> published;
    std::vector<zeno_water_device::Launch> directory;
    zeno_water_device::Launch* deviceDirectory=nullptr;
    zeno_water_device::Launch* upload=nullptr;
    cudaEvent_t copied{};
    bool haveCopy=false;
    uint64_t slots=0,nextVersion=0;
    Statistics stats;
    Impl() {
        try {
            check(cudaHostAlloc(reinterpret_cast<void**>(&upload),sizeof(*upload)*zeno_water_device::MaxSceneSurfaces,cudaHostAllocDefault));
            check(cudaMallocAsync(reinterpret_cast<void**>(&deviceDirectory),sizeof(*deviceDirectory)*zeno_water_device::MaxSceneSurfaces,0));
            check(cudaEventCreateWithFlags(&copied,cudaEventDisableTiming));
        } catch(...) {
            if(upload)cudaFreeHost(upload);
            if(deviceDirectory)cudaFreeAsync(deviceDirectory,0);
            if(copied)cudaEventDestroy(copied);
            throw;
        }
    }
    ~Impl() {
        sources.clear();
        if(haveCopy)cudaEventSynchronize(copied);
        if(upload)cudaFreeHost(upload);
        if(deviceDirectory)cudaFreeAsync(deviceDirectory,0);
        if(copied)cudaEventDestroy(copied);
    }
    unsigned allocateSlot() {
        for(unsigned slot=0;slot<zeno_water_device::MaxSceneSurfaces;++slot)if(!(slots&(1ull<<slot))) {
            slots|=1ull<<slot;return slot;
        }
        throw std::runtime_error("Too many independent water surfaces (maximum 64)");
    }
};
SceneRuntime::SceneRuntime():impl(std::make_unique<Impl>()){}
SceneRuntime::~SceneRuntime()=default;
bool SceneRuntime::update(OptixDeviceContext context,uint32_t maxSbt,const std::vector<SceneSurface>& surfaces,View view,bool interactive) {
    const auto begin=std::chrono::steady_clock::now();auto& p=*impl;bool changed=false;
    std::set<std::string> live;
    for(const auto& input:surfaces)if(!live.insert(input.key).second)throw std::runtime_error("Duplicate water resource: "+input.key);
    for(auto it=p.sources.begin();it!=p.sources.end();) {
        if(live.count(it->first)){++it;continue;}
        p.slots&=~(1ull<<it->second.slot);it=p.sources.erase(it);changed=true;
    }
    Statistics total{};total.minDepth=~0u;total.reusedTopology=true;
    for(const auto& input:surfaces) {
        auto inserted=p.sources.try_emplace(input.key);auto& source=inserted.first->second;
        if(inserted.second){source.slot=p.allocateSlot();source.version=++p.nextVersion;changed=true;}
        bool sourceChanged=source.visible!=input.visible;source.visible=input.visible;
        if(input.visible) {
            if(!source.runtime)source.runtime=std::make_unique<Runtime>();
            auto settings=input.settings;settings.clusterIdBase=source.slot*zeno_water_device::ObjectClusterSpan;
            source.runtime->update(context,maxSbt,{settings},view,interactive);
            const auto s=source.runtime->statistics();sourceChanged|=source.nativeVersion!=s.revision;
            if(source.nativeVersion!=s.revision) {
                total.fftMs+=s.fftMs;total.tessellationMs+=s.tessellationMs;total.clasMs+=s.clasMs;total.gasMs+=s.gasMs;
                total.reusedTopology&=s.reusedTopology;
            }
            source.nativeVersion=s.revision;total.clusters+=s.clusters;total.triangles+=s.triangles;
            total.topologyRevision+=s.topologyRevision;
            total.minDepth=std::min(total.minDepth,s.minDepth);total.maxDepth=std::max(total.maxDepth,s.maxDepth);
            total.effectiveEdgePixels=std::max(total.effectiveEdgePixels,s.effectiveEdgePixels);
            total.pending|=s.pending;total.qualityLimited|=s.qualityLimited;
        }
        if(source.runtime)total.bytes+=source.runtime->statistics().bytes;
        if(sourceChanged){source.version=++p.nextVersion;changed=true;}
    }
    std::vector<zeno_water_device::Launch> directory;
    p.published.clear();p.published.reserve(p.sources.size());
    for(const auto& [key,source]:p.sources) {
        if(directory.size()<=source.slot)directory.resize(source.slot+1);
        const auto handle=source.visible && source.runtime?source.runtime->handle():0;
        if(handle)directory[source.slot]=source.runtime->launch();
        p.published.push_back({key,handle,source.version});
    }
    const bool directoryChanged=directory.size()!=p.directory.size() ||
        !std::equal(directory.begin(),directory.end(),p.directory.begin(),equal);
    if(directoryChanged) {
        if(p.haveCopy)check(cudaEventSynchronize(p.copied));
        std::copy(directory.begin(),directory.end(),p.upload);
        if(!directory.empty())check(cudaMemcpyAsync(p.deviceDirectory,p.upload,directory.size()*sizeof(*p.upload),cudaMemcpyHostToDevice,0));
        check(cudaEventRecord(p.copied,0));p.haveCopy=true;p.directory=std::move(directory);
    }
    total.bytes+=sizeof(zeno_water_device::Launch)*zeno_water_device::MaxSceneSurfaces;
    total.revision=p.stats.revision+(changed?1:0);if(total.minDepth==~0u)total.minDepth=0;
    total.submitMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    p.stats=total;return changed;
}
void SceneRuntime::rendered(cudaStream_t stream,bool auxiliary) {
    for(auto& [_,source]:impl->sources)if(source.visible && source.runtime)source.runtime->rendered(stream,auxiliary);
}
zeno_water_device::SceneLaunch SceneRuntime::launch() const {return {impl->deviceDirectory,unsigned(impl->directory.size())};}
const std::vector<PublishedSurface>& SceneRuntime::publications() const {return impl->published;}
Statistics SceneRuntime::statistics() const {return impl->stats;}
} // namespace zeno::water
