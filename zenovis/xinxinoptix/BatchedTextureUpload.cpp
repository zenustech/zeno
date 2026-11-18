#include "BatchedTextureUpload.h"
#include "ClusterProfiling.h"
#include "TextureSyncEvent.h"
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <zeno/utils/FrameCapture.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace zeno { namespace cluster { namespace {
using Clock=std::chrono::steady_clock;
bool enabled(){static const bool value=[] {const auto* p=std::getenv("ZENO_CLAS_TEXTURE_BATCH_MAP");return !p||*p!='0';}();return value;}
unsigned int batchLimit(){static const unsigned int value=[] {const auto* p=std::getenv("ZENO_CLAS_TEXTURE_MAP_BATCH_SIZE");return p?std::clamp(std::atoi(p),1,32):8;}();return value;}
unsigned int gpuLimit(){static const unsigned int value=[] {const auto* p=std::getenv("ZENO_CLAS_TEXTURE_GPU_INFLIGHT");return p?unsigned(std::clamp(std::atoi(p),0,128)):0u;}();return value;}
const char* mappingDirectory(){static const auto* value=std::getenv("ZENO_CLAS_TEXTURE_TIMELINE");return value&&*value?value:nullptr;}
bool recordMappings(){return framecapture::enabled()||mappingDirectory();}
struct Transfer {
    void* owner{};
    unsigned int page{};
    CUcontext context{};
    CUstream stream{};
    std::vector<CUarrayMapInfo> maps;
    std::vector<CUDA_MEMCPY2D> copies;
    std::vector<std::function<void()>> after;
};
thread_local std::unique_ptr<Transfer> capture;
thread_local std::unique_ptr<Transfer> unmaps;
struct PreparedBatch {void* owner;CUstream stream;CUcontext context;std::vector<std::unique_ptr<Transfer>> jobs;std::vector<CUarrayMapInfo> maps;};
thread_local PreparedBatch preparedStorage;
thread_local PreparedBatch* preparedBatch=nullptr;
struct PreparedTransferPool {
    std::mutex mutex;
    std::vector<std::unique_ptr<Transfer>> free;
    PreparedTransferPool(){free.reserve(16);}
    std::unique_ptr<Transfer> acquire() {
        std::lock_guard<std::mutex> lock(mutex);
        if(!free.empty()){auto t=std::move(free.back());free.pop_back();return t;}
        auto t=std::make_unique<Transfer>();t->maps.reserve(4);t->copies.reserve(16);t->after.reserve(4);return t;
    }
    void release(std::unique_ptr<Transfer> t) {
        t->maps.clear();t->copies.clear();t->after.clear();
        std::lock_guard<std::mutex> lock(mutex);if(free.size()<16)free.push_back(std::move(t));
    }
};
thread_local std::shared_ptr<PreparedTransferPool> preparedTransfers;
struct MappingTrace {
    std::mutex mutex;
    std::unique_ptr<framecapture::Recorder> recorder;
    unsigned int generation=0;
};
MappingTrace& mappingTrace(){static MappingTrace value;return value;}
struct UploadStreams {
    struct Stream {CUcontext context;CUstream stream;};
    std::mutex mutex;
    std::map<void*,Stream> streams;
};
UploadStreams& uploadStreams(){static UploadStreams value;return value;}
CUresult submitMaps(CUarrayMapInfo* maps,unsigned int count,CUstream stream) {
    if(!count)return CUDA_SUCCESS;
    // Keep each driver submission within one array, mip and backing heap,
    // matching the resource grouping used by RTXTS. In particular, avoid a
    // heterogeneous list of BC array bindings while textures are sampled.
    using Key=std::tuple<CUresourcetype,uintptr_t,unsigned int,CUmemGenericAllocationHandle>;
    std::map<Key,std::vector<CUarrayMapInfo>> groups;
    auto key=[](const CUarrayMapInfo& m)->Key {
        const auto resource=m.resourceType==CU_RESOURCE_TYPE_MIPMAPPED_ARRAY?
            reinterpret_cast<uintptr_t>(m.resource.mipmap):reinterpret_cast<uintptr_t>(m.resource.array);
        const auto level=m.subresourceType==CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL?m.subresource.sparseLevel.level:~0u;
        return {m.resourceType,resource,level,m.memHandle.memHandle};
    };
    bool homogeneous=true;
    for(unsigned int i=1;i<count;++i)if(key(maps[i])!=key(maps[0])){homogeneous=false;break;}
    if(count>1&&!homogeneous) {
        for(unsigned int i=0;i<count;++i) {
            const auto& m=maps[i];
            const auto resource=m.resourceType==CU_RESOURCE_TYPE_MIPMAPPED_ARRAY?
                reinterpret_cast<uintptr_t>(m.resource.mipmap):reinterpret_cast<uintptr_t>(m.resource.array);
            const auto level=m.subresourceType==CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL?m.subresource.sparseLevel.level:~0u;
            groups[{m.resourceType,resource,level,m.memHandle.memHandle}].push_back(m);
        }
        if(groups.size()>1) {
            for(auto& group:groups){const auto r=submitMaps(group.second.data(),unsigned(group.second.size()),stream);if(r!=CUDA_SUCCESS)return r;}
            return CUDA_SUCCESS;
        }
    }
    const auto start=framecapture::nowMs();
    const bool unmapping=maps[0].memOperationType==CU_MEM_OPERATION_TYPE_UNMAP;
    TimelineRange apiRange(unmapping?"Textures: sparse unmap":"Textures: sparse map",count);
    beforeTextureMapping(stream);
    CUresult result;
    {
        result=cuMemMapArrayAsync(maps,count,stream);
    }
    if(result==CUDA_SUCCESS)afterTextureMapping(stream);
    if(recordMappings()) {
        unsigned int unmapped=0;for(unsigned int i=0;i<count;++i)unmapped+=maps[i].memOperationType==CU_MEM_OPERATION_TYPE_UNMAP;
        const auto elapsed=framecapture::nowMs()-start;
        auto& trace=mappingTrace();std::lock_guard<std::mutex> lock(trace.mutex);
        if(!trace.recorder) {
            const auto name="texture-mapping-"+std::to_string(trace.generation++);
            if(const auto* directory=mappingDirectory())trace.recorder=std::make_unique<framecapture::Recorder>(
                std::filesystem::u8path(directory),name.c_str(),"entries,unmaps",true);
            else trace.recorder=std::make_unique<framecapture::Recorder>(name.c_str(),"entries,unmaps");
        }
        trace.recorder->record(start,elapsed,{double(count),double(unmapped)});
    }
    return result;
}
class Transfers {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::unique_ptr<Transfer>> queue;
    std::map<void*,std::set<unsigned int>> pending;
    std::thread worker;
    bool stopping=false,draining=false;
    void run() noexcept {
        try {
            for(;;) {
                std::vector<std::unique_ptr<Transfer>> jobs;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    wake.wait(lock,[&]{return stopping||!queue.empty();});
                    if(queue.empty())return;
                    // A short bounded collection window also groups tiles
                    // decoded simultaneously by different reader threads.
                    const auto deadline=Clock::now()+std::chrono::microseconds(250);
                    while(queue.size()<batchLimit()&&!stopping)
                        if(wake.wait_until(lock,deadline)==std::cv_status::timeout)break;
                    const auto owner=queue.front()->owner;
                    const auto context=queue.front()->context;
                    const auto stream=queue.front()->stream;
                    unsigned int limit=batchLimit();
                    if(const auto maximum=gpuLimit()) {
                        const auto submitted=submittedTextureUploads(owner);
                        if(submitted>=maximum) {wake.wait_for(lock,std::chrono::milliseconds(1));continue;}
                        limit=std::min(limit,maximum-submitted);
                    }
                    while(!queue.empty()&&jobs.size()<limit&&queue.front()->owner==owner&&
                          queue.front()->context==context&&queue.front()->stream==stream) {
                        jobs.push_back(std::move(queue.front()));queue.pop_front();
                    }
                }
                OTK_ERROR_CHECK(zeno::cluster::setTextureContext(jobs.front()->context));
                TimelineRange batchRange("Textures: upload batch",jobs.size());
                std::vector<CUarrayMapInfo> maps;
                for(const auto& job:jobs)maps.insert(maps.end(),job->maps.begin(),job->maps.end());
                OTK_ERROR_CHECK(submitMaps(maps.data(),unsigned(maps.size()),jobs.front()->stream));
                for(auto& job:jobs) {
                    for(const auto& copy:job->copies)OTK_ERROR_CHECK(cuMemcpy2DAsync(&copy,job->stream));
                    if(recycleCompletedTextureTransfers()) {
                        // One completion owns publication and staging release.
                        // The pinned pool needs no second GPU event for bytes
                        // whose copy this completion already waited for.
                        enqueueTextureUpload(job->stream,job->owner,job->page,
                            [actions=std::move(job->after)]() mutable {for(auto& action:actions)action();},true);
                    } else for(auto& action:job->after)action();
                    std::lock_guard<std::mutex> lock(mutex);
                    auto it=pending.find(job->owner);it->second.erase(job->page);
                    if(it->second.empty())pending.erase(it);
                }
                wake.notify_all();
            }
        } catch(const std::exception& e) {
            std::fprintf(stderr,"OTK batched texture transfer failed: %s\n",e.what());std::terminate();
        }
    }
public:
    ~Transfers(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        wake.notify_all();if(worker.joinable())worker.join();
    }
    void enqueue(std::unique_ptr<Transfer> job) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[&]{return !draining;});
        if(!pending[job->owner].insert(job->page).second)throw std::logic_error("Duplicate batched texture upload");
        queue.push_back(std::move(job));
        if(!worker.joinable())worker=std::thread([this]{run();});
        wake.notify_all();
    }
    bool contains(void* owner,unsigned int page){std::lock_guard<std::mutex> lock(mutex);const auto i=pending.find(owner);return i!=pending.end()&&i->second.count(page);}
    unsigned int count(void* owner){std::lock_guard<std::mutex> lock(mutex);const auto i=pending.find(owner);return i==pending.end()?0u:unsigned(i->second.size());}
    void drain(void* owner){
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock,[&]{return !draining&&!pending.count(owner);});
        if(!pending.empty())return;
        draining=true;stopping=true;lock.unlock();wake.notify_all();
        if(worker.joinable())worker.join();
        lock.lock();stopping=false;draining=false;lock.unlock();wake.notify_all();
    }
};
Transfers& transfers(){static Transfers value;return value;}
} // namespace
bool beginBatchedTextureTransfer(void* owner,unsigned int page,CUstream stream,bool newStorage) {
    if(!enabled()||!newStorage||(usePreparedTextureUploads(owner)&&!preparedBatch))return false;
    if(capture||unmaps)throw std::logic_error("Nested batched texture transfer");
    capture=preparedBatch?preparedTransfers->acquire():std::make_unique<Transfer>();
    capture->owner=owner;capture->page=page;capture->stream=stream;
    if(preparedBatch)capture->context=preparedBatch->context;
    else OTK_ERROR_CHECK(cuCtxGetCurrent(&capture->context));return true;
}
void endBatchedTextureTransfer(){
    if(preparedBatch)preparedBatch->jobs.push_back(std::move(capture));
    else transfers().enqueue(std::move(capture));
}
void beginPreparedTextureBatch(void* owner,CUstream stream) {
    if(preparedBatch||capture||unmaps)throw std::logic_error("Nested prepared texture batch");
    if(!preparedTransfers)preparedTransfers=std::make_shared<PreparedTransferPool>();
    preparedBatch=&preparedStorage;preparedBatch->owner=owner;preparedBatch->stream=stream;
    preparedBatch->jobs.clear();preparedBatch->jobs.reserve(4);
    preparedBatch->maps.clear();preparedBatch->maps.reserve(16);
    OTK_ERROR_CHECK(cuCtxGetCurrent(&preparedBatch->context));
}
void endPreparedTextureBatch() {
    auto* batch=preparedBatch;preparedBatch=nullptr;
    auto& maps=batch->maps;
    for(const auto& job:batch->jobs) {
        if(job->owner!=batch->owner||job->stream!=batch->stream)throw std::logic_error("Prepared batch owner changed");
        maps.insert(maps.end(),job->maps.begin(),job->maps.end());
    }
    OTK_ERROR_CHECK(submitMaps(maps.data(),unsigned(maps.size()),batch->stream));
    for(auto& job:batch->jobs) {
        for(const auto& copy:job->copies)OTK_ERROR_CHECK(cuMemcpy2DAsync(&copy,batch->stream));
        if(recycleCompletedTextureTransfers()) {
            auto* transfer=job.release();
            enqueueTextureUpload(batch->stream,transfer->owner,transfer->page,
                [transfer,pool=preparedTransfers]() mutable {
                    for(auto& action:transfer->after)action();
                    pool->release(std::unique_ptr<Transfer>(transfer));
                },true);
        } else {
            for(auto& action:job->after)action();
            preparedTransfers->release(std::move(job));
        }
    }
    batch->jobs.clear();
}
bool capturingTextureTransfer(){return bool(capture);}
bool capturingTextureUnmaps(){return bool(unmaps);}
void deferTextureTransferAction(std::function<void()> action){capture->after.push_back(std::move(action));}
CUresult mapTextureArray(CUarrayMapInfo* maps,unsigned int count,CUstream stream){
    auto* job=capture?capture.get():unmaps.get();
    if(job){if(job->stream!=stream)return CUDA_ERROR_INVALID_VALUE;job->maps.insert(job->maps.end(),maps,maps+count);return CUDA_SUCCESS;}
    return submitMaps(maps,count,stream);
}
CUresult copyTextureTile(const CUDA_MEMCPY2D* copy,CUstream stream){
    if(capture){if(capture->stream!=stream)return CUDA_ERROR_INVALID_VALUE;capture->copies.push_back(*copy);return CUDA_SUCCESS;}
    return cuMemcpy2DAsync(copy,stream);
}
bool batchedTextureTransferPending(void* owner,unsigned int page){return enabled()&&transfers().contains(owner,page);}
unsigned int pendingBatchedTextureTransfers(void* owner){return enabled()?transfers().count(owner):0;}
void drainBatchedTextureTransfers(void* owner){if(enabled())transfers().drain(owner);}
void releaseTextureMappingRecorder(){
    if(!recordMappings())return;
    // A Windows DLL TLS destructor runs under the loader lock. Joining a
    // diagnostic writer there can deadlock its thread's DLL detach. Drain
    // this ordinary-owned writer explicitly while the loader is still alive.
    auto& trace=mappingTrace();std::lock_guard<std::mutex> lock(trace.mutex);trace.recorder.reset();
}
CUstream textureUploadStream(void* owner,CUstream requestStream){
    if(!enabled()||usePreparedTextureUploads(owner))return requestStream;
    auto& state=uploadStreams();std::lock_guard<std::mutex> lock(state.mutex);
    const auto found=state.streams.find(owner);if(found!=state.streams.end())return found->second.stream;
    UploadStreams::Stream created{};OTK_ERROR_CHECK(cuCtxGetCurrent(&created.context));
    OTK_ERROR_CHECK(cuStreamCreate(&created.stream,CU_STREAM_NON_BLOCKING));
    state.streams.emplace(owner,created);return created.stream;
}
void releaseTextureUploadStream(void* owner){
    auto& state=uploadStreams();UploadStreams::Stream retired{};
    {std::lock_guard<std::mutex> lock(state.mutex);const auto it=state.streams.find(owner);
        if(it==state.streams.end())return;retired=it->second;state.streams.erase(it);}
    OTK_ERROR_CHECK(cuCtxPushCurrent(retired.context));
    OTK_ERROR_CHECK(cuStreamSynchronize(retired.stream));
    OTK_ERROR_CHECK(cuStreamDestroy(retired.stream));
    CUcontext previous{};OTK_ERROR_CHECK(cuCtxPopCurrent(&previous));
}
bool beginBatchedTextureUnmaps(void* owner,CUstream stream){
    if(!enabled())return false;
    if(capture||unmaps)throw std::logic_error("Nested batched texture unmapping");
    unmaps=std::make_unique<Transfer>();unmaps->owner=owner;unmaps->stream=stream;return true;
}
void endBatchedTextureUnmaps(){
    auto job=std::move(unmaps);
    const auto start=framecapture::nowMs();
    OTK_ERROR_CHECK(submitMaps(job->maps.data(),unsigned(job->maps.size()),job->stream));
    if(!job->maps.empty())noteTextureEviction(job->owner,framecapture::nowMs()-start,unsigned(job->maps.size()));
    for(auto& release:job->after)release();
}
void afterTextureUnmaps(std::function<void()> release){if(unmaps)unmaps->after.push_back(std::move(release));else release();}
bool retainTextureStandby(){static const bool value=[] {const auto* p=std::getenv("ZENO_CLAS_TEXTURE_STANDBY");return !p||*p!='0';}();return value;}
bool recycleCompletedTextureTransfers(){static const bool value=[] {const auto* p=std::getenv("ZENO_CLAS_TEXTURE_COMPLETION_RECYCLE");return !p||*p!='0';}();return enabled()&&value;}
}} // namespace zeno::cluster
