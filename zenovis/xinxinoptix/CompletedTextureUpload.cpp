#include "CompletedTextureUpload.h"
#include "BatchedTextureUpload.h"
#include "DeferredTextureCallbacks.h"
#include "BackgroundEventWait.h"
#include "TextureSyncEvent.h"
#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include <zeno/utils/FrameCapture.h>
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace zeno { namespace cluster { namespace {
struct Upload {
    CUcontext context{};
    CUevent completed{};
    void* owner{};
    unsigned int page{};
    bool tile=false;
    std::function<void()> publish;
    ~Upload() { if(completed) OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(completed)); }
};
class Completions {
    struct CachedEvent {CUcontext context;CUevent event;};
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::unique_ptr<Upload>> queue;
    std::map<void*,std::set<unsigned int>> pending;
    std::map<void*,CompletedTextureStats> counts;
    std::map<void*,std::vector<CachedEvent>> events;
    std::thread worker;
    bool stopping=false,draining=false;
    static bool reuseEvents() {
        static const bool reuse=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_EVENT_REUSE");return !v||*v!='0';}();
        return reuse;
    }
    void run() noexcept {
        try {
            for(;;) {
                std::unique_ptr<Upload> job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock,[&]{return stopping||!queue.empty();});
                    if(queue.empty())return;
                    job=std::move(queue.front());queue.pop_front();
                }
                OTK_ERROR_CHECK(setTextureContext(job->context));
                // Only this completion worker waits. IO workers can decode
                // and queue subsequent uploads while rendering continues.
                OTK_ERROR_CHECK(waitBackgroundEvent(job->completed));
                job->publish();
                const auto owner=job->owner;const auto page=job->page;const bool tile=job->tile;
                // Only a completed, published event is recycled. No reader
                // outside this worker observes its handle or waits on it.
                if(reuseEvents()) {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto& pool=events[owner];
                    if(pool.size()<64){pool.push_back({job->context,job->completed});job->completed=nullptr;}
                }
                job.reset();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto it=pending.find(owner);
                    if(tile)++counts[owner].tiles;else ++counts[owner].samplers;
                    it->second.erase(page);
                    if(it->second.empty())pending.erase(it);
                }
                wake.notify_all();
                wakePreparedTextureUploads(owner);
            }
        } catch(const std::exception& error) {
            std::fprintf(stderr,"OTK upload completion failed: %s\n",error.what());
            std::terminate();
        }
    }
public:
    ~Completions() {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[&]{return !draining&&pending.empty();});
        stopping=true;lock.unlock();wake.notify_all();
        if(worker.joinable())worker.join();
    }
    void enqueue(std::unique_ptr<Upload> job) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[&]{return !draining;});
        if(!pending[job->owner].insert(job->page).second)
            throw std::runtime_error("Duplicate in-flight texture upload");
        queue.push_back(std::move(job));
        if(!worker.joinable())worker=std::thread([this]{run();});
        wake.notify_all();
    }
    CUevent acquire(void* owner,CUcontext context) {
        if(reuseEvents()) {
            std::lock_guard<std::mutex> lock(mutex);auto& pool=events[owner];
            for(auto it=pool.begin();it!=pool.end();++it)if(it->context==context) {
                const auto event=it->event;pool.erase(it);++counts[owner].eventsReused;return event;
            }
        }
        CUevent event{};OTK_ERROR_CHECK(cuEventCreate(&event,CU_EVENT_DISABLE_TIMING));
        {std::lock_guard<std::mutex> lock(mutex);++counts[owner].eventsCreated;}
        return event;
    }
    bool contains(void* owner,unsigned int page) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it=pending.find(owner);
        return it!=pending.end()&&it->second.count(page)!=0;
    }
    unsigned int count(void* owner) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it=pending.find(owner);
        return it==pending.end()?0u:static_cast<unsigned int>(it->second.size());
    }
    CompletedTextureStats stats(void* owner) {
        std::lock_guard<std::mutex> lock(mutex);const auto it=counts.find(owner);
        auto result=it==counts.end()?CompletedTextureStats{}:it->second;
        const auto e=events.find(owner);if(e!=events.end())result.cachedEvents=e->second.size();return result;
    }
    void evicted(void* owner,double milliseconds,unsigned int count) {
        std::lock_guard<std::mutex> lock(mutex);auto& s=counts[owner];
        s.evictions+=count;s.evictionMs+=milliseconds;if(milliseconds>s.maxEvictionMs)s.maxEvictionMs=milliseconds;
    }
    void recycled(void* owner){std::lock_guard<std::mutex> lock(mutex);++counts[owner].recycledBuffers;}
    void forget(void* owner) {
        std::vector<CachedEvent> retired;
        {std::lock_guard<std::mutex> lock(mutex);counts.erase(owner);const auto it=events.find(owner);
            if(it!=events.end()){retired=std::move(it->second);events.erase(it);}}
        // Loader teardown drains its tasks before this call, while its CUDA
        // context is still alive. Retire cached events here, not at exit.
        for(const auto& e:retired) {
            const auto pushed=cuCtxPushCurrent(e.context);
            if(pushed==CUDA_SUCCESS) {
                OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(e.event));
                CUcontext old{};OTK_ERROR_CHECK_NOTHROW(cuCtxPopCurrent(&old));
            } else OTK_ERROR_CHECK_NOTHROW(pushed);
        }
    }
    void drain(void* owner) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[&]{return !draining&&pending.find(owner)==pending.end();});
        if(!pending.empty())return;
        // Retire the CUDA-bound worker before the last loader/context goes
        // away. Enqueueing for another loader resumes after the join.
        draining=true;stopping=true;lock.unlock();wake.notify_all();
        if(worker.joinable())worker.join();
        lock.lock();stopping=false;draining=false;lock.unlock();wake.notify_all();
    }
};
Completions& completions() { static Completions queue;return queue; }
std::unique_ptr<framecapture::Recorder>& pagingRecorder() {
    static thread_local std::unique_ptr<framecapture::Recorder> recorder;
    return recorder;
}
} // namespace
void enqueueTextureUpload(CUstream stream,void* owner,unsigned int page,std::function<void()> publish,bool tile) {
    if(capturingTextureTransfer()) {
        if(recycleCompletedTextureTransfers())deferTextureTransferAction(std::move(publish));
        else deferTextureTransferAction([=,publish=std::move(publish)]() mutable {enqueueTextureUpload(stream,owner,page,std::move(publish),tile);});
        return;
    }
    auto job=std::unique_ptr<Upload>(new Upload);
    job->owner=owner;job->page=page;job->publish=std::move(publish);
    job->tile=tile;
    OTK_ERROR_CHECK(cuCtxGetCurrent(&job->context));
    job->completed=completions().acquire(owner,job->context);
    OTK_ERROR_CHECK(cuEventRecord(job->completed,stream));
    completions().enqueue(std::move(job));
}
bool textureUploadPending(void* owner,unsigned int page) {
    // A batched transfer registers its completion before dropping its own
    // pending marker. Inspect the earlier stage first so a concurrent handoff
    // cannot make a still-uploading page appear absent from both stages.
    return batchedTextureTransferPending(owner,page)||completions().contains(owner,page)||preparedTextureUploadPending(owner,page);
}
unsigned int pendingTextureUploads(void* owner) {
    const auto preparing=pendingBatchedTextureTransfers(owner);
    return preparing+completions().count(owner);
}
unsigned int submittedTextureUploads(void* owner) {return completions().count(owner);}
void drainTextureUploads(void* owner) { drainBatchedTextureTransfers(owner);completions().drain(owner);releaseTextureMappingRecorder(); }
CompletedTextureStats completedTextureStats(void* owner){return completions().stats(owner);}
uint64_t textureTileBackingBytes(void* owner) {
    auto* loader=static_cast<demandLoading::DemandLoaderImpl*>(static_cast<demandLoading::DemandLoader*>(owner));
    return loader->getDeviceMemoryManager()->getTextureTileMemory();
}
void forgetTextureUploadStatistics(void* owner){
    releaseTextureUploadStream(owner);
    if(textureWorkDiagnosticsEnabled())std::fprintf(stderr,"[CLAS texture transfers] recycled staging buffers=%llu\n",
        static_cast<unsigned long long>(completions().stats(owner).recycledBuffers));
    completions().forget(owner);releaseTextureMappingRecorder();
}
void noteTextureTransferRecycle(void* owner){if(textureWorkDiagnosticsEnabled())completions().recycled(owner);}
void noteTextureEviction(void* owner,double milliseconds,unsigned int count) {
    if(textureWorkDiagnosticsEnabled()&&!capturingTextureUnmaps())completions().evicted(owner,milliseconds,count);
}
bool textureWorkDiagnosticsEnabled(){return framecapture::enabled();}
void noteTexturePaging(unsigned int lists,unsigned int empty,unsigned int pages,unsigned int requested,
                       unsigned int stale,bool evictionActive,double milliseconds,
                       unsigned int restored,unsigned int invalidated,bool standby) {
    if(!textureWorkDiagnosticsEnabled())return;
    auto& recorder=pagingRecorder();
    if(!recorder)recorder=std::make_unique<framecapture::Recorder>("paging","lists,empty_lists,staged_pages,requests,stale_pages,eviction_active,restored,invalidated,standby_policy");
    recorder->record(framecapture::nowMs()-milliseconds,milliseconds,
        {double(lists),double(empty),double(pages),double(requested),double(stale),double(evictionActive),double(restored),double(invalidated),double(standby)});
}
void releaseTexturePagingRecorder(){pagingRecorder().reset();}
TextureCallbackStats textureCallbackStats(){return demandLoading::CudaCallback::stats();}
}} // namespace zeno::cluster
