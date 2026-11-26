#include "CompletedTextureUpload.h"
#include "BatchedTextureUpload.h"
#include "ClusterProfiling.h"
#include "BackgroundEventWait.h"
#include "TextureUploadSchedule.h"
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace zeno { namespace cluster { namespace {
using Clock=std::chrono::steady_clock;
bool nonblockingMapCredits() {
    static const bool enabled=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_NONBLOCKING_MAP_CREDITS");return !v||*v!='0';}();
    return enabled;
}
bool inlineBatches() {
    static const bool enabled=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_INLINE_BATCH");return !v||*v!='0';}();
    return enabled;
}
struct Job {
    void* owner{};unsigned int page{};
    std::vector<char> data;
    std::function<void(CUstream)> upload;
    Clock::time_point ready=Clock::now();
    uint64_t resourceGroup=~uint64_t(0);
};
thread_local const Job* activeJob=nullptr;
struct MappingBudget {
    CUstream stream;
    std::array<CUevent,4> events{};
    unsigned int next=0,count=0;
    explicit MappingBudget(CUstream s):stream(s) {
        for(auto& event:events)OTK_ERROR_CHECK(cuEventCreate(&event,CU_EVENT_DISABLE_TIMING));
    }
    ~MappingBudget(){for(auto event:events)OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(event));}
    bool canStart() {
        if(count<events.size())return true;
        const auto status=cuEventQuery(events[next]);
        if(status==CUDA_ERROR_NOT_READY)return false;
        OTK_ERROR_CHECK(status);return true;
    }
    void before(CUstream s) {
        if(s!=stream)throw std::logic_error("Prepared mapping stream changed");
        // Admission already checked a credit before taking SDK locks or
        // becoming active. Keep one SDK job atomic, including its bounded
        // eviction fanout; never wait for a GPU credit inside that job.
        if(nonblockingMapCredits())return;
        if(count<events.size())return;
        const auto status=cuEventQuery(events[next]);
        if(status==CUDA_ERROR_NOT_READY) {
            TimelineRange wait("Textures: mapping credit wait");
            OTK_ERROR_CHECK(waitBackgroundEvent(events[next]));
        } else OTK_ERROR_CHECK(status);
    }
    void after(CUstream s) {
        if(s!=stream)throw std::logic_error("Prepared mapping stream changed");
        OTK_ERROR_CHECK(cuEventRecord(events[next],stream));
        next=(next+1)%events.size();count=std::min(unsigned(events.size()),count+1);
    }
};
thread_local MappingBudget* activeMappingBudget=nullptr;
class PreparedUploads {
    void* owner;
    CUcontext context{};
    CUevent phaseReady{};
    uint64_t lastBarrierSubmission=0;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Job> jobs;
    std::set<unsigned int> pages;
    std::thread worker;
    bool stopping=false,unlimited=false,active=false;
    unsigned int allowance=0;
    uint64_t byteAllowance=0;
    double timeAllowance=0;
    double stationaryBudget=2.;
    double submissionWindowMs=8.;
    Clock::time_point windowDeadline{};
    PreparedTextureStats stats{};
    TextureUploadPriorities priorities;
    auto selectJob(unsigned int offset=0,uint64_t resourceGroup=~uint64_t(0)) {
        auto best=jobs.end();uint64_t highest=0;
        for(auto it=jobs.begin();it!=jobs.end();++it) {
            if(!unlimited&&it->data.size()>byteAllowance)continue;
            if(resourceGroup!=~uint64_t(0)&&it->resourceGroup!=resourceGroup)continue;
            // One in eight submissions preserves progress for older views.
            if((stats.submitted+offset)%8==7)return it;
            const auto score=priorities.score(it->page,it->data.empty());
            if(best==jobs.end()||score>highest){best=it;highest=score;}
        }
        return best;
    }
    void run() noexcept {
        CUstream stream{};
        struct Timer {CUevent begin{},end{};};
        std::vector<Timer> timers;
        std::vector<unsigned int> freeTimers;
        std::deque<unsigned int> pendingTimers;
        try {
            OTK_ERROR_CHECK(cuCtxSetCurrent(context));
            OTK_ERROR_CHECK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
            std::unique_ptr<MappingBudget> mappingBudget;
            if(submissionWindowMs>0)mappingBudget=std::make_unique<MappingBudget>(stream);
            const auto* timing=std::getenv("ZENO_OPTIX_RENDER_TIMING");
            if(timing && timing[0]=='1')for(unsigned int i=0;i<64;++i) {
                Timer timer;OTK_ERROR_CHECK(cuEventCreate(&timer.begin,CU_EVENT_DEFAULT));
                OTK_ERROR_CHECK(cuEventCreate(&timer.end,CU_EVENT_DEFAULT));
                timers.push_back(timer);freeTimers.push_back(i);
            }
            auto pollTimers=[&] {
                while(!pendingTimers.empty()) {
                    const auto index=pendingTimers.front();const auto status=cuEventQuery(timers[index].end);
                    if(status==CUDA_ERROR_NOT_READY)break;
                    OTK_ERROR_CHECK(status);float ms=0;
                    OTK_ERROR_CHECK(cuEventElapsedTime(&ms,timers[index].begin,timers[index].end));
                    {std::lock_guard<std::mutex> lock(mutex);stats.gpuMs+=ms;stats.maxGpuMs=std::max(stats.maxGpuMs,double(ms));}
                    pendingTimers.pop_front();freeTimers.push_back(index);
                }
            };
            std::vector<Job> batch;batch.reserve(4);
            static const bool recordSlow=std::getenv("ZENO_OPTIX_CAMERA_RECORD")!=nullptr;
            for(;;) {
                pollTimers();
                batch.clear();
                Clock::time_point batchDeadline{};bool batchUnlimited=false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    // Completion events are retired by a different worker.
                    // A bounded in-flight queue prevents an upload backlog on
                    // WDDM while the CPU can retain more prepared BC tiles.
                    auto runnable=[&] {
                        return stopping || (!jobs.empty() && (unlimited ||
                            (allowance && timeAllowance>0 && selectJob()!=jobs.end() &&
                             (!nonblockingMapCredits()||submissionWindowMs<=0||Clock::now()<windowDeadline))));
                    };
                    if(pendingTimers.empty())wake.wait(lock,runnable);
                    else if(!wake.wait_for(lock,std::chrono::milliseconds(1),runnable))continue;
                    if(stopping && jobs.empty())break;
                    if(mappingBudget&&nonblockingMapCredits()) {
                        // A GPU wait inside active=true also stalls the render
                        // thread in pause(). Poll before admission, with no
                        // queue/SDK mutex held, then recheck the revoked window.
                        lock.unlock();
                        if(!mappingBudget->canStart()){pauseBackgroundEventQuery();continue;}
                        lock.lock();
                        if(!runnable())continue;
                        if(stopping&&jobs.empty())break;
                    }
                    const auto inFlightLimit=submissionWindowMs>0?8u:2u;
                    if(!unlimited && pendingTextureUploads(owner)>=inFlightLimit) {
                        wake.wait_for(lock,std::chrono::milliseconds(1),[&]{return stopping||unlimited||!allowance||pendingTextureUploads(owner)<inFlightLimit;});continue;
                    }
                    const auto pending=pendingTextureUploads(owner);
                    const auto limit=std::min(inlineBatches()?4u:1u,unlimited?4u:(pending<inFlightLimit?inFlightLimit-pending:0u));
                    while(batch.size()<limit&&(unlimited||allowance)) {
                        if(!batch.empty()&&batch.front().resourceGroup==~uint64_t(0))break;
                        const auto next=selectJob(unsigned(batch.size()),batch.empty()?~uint64_t(0):batch.front().resourceGroup);
                        if(next==jobs.end())break;
                        batch.push_back(std::move(*next));jobs.erase(next);
                        stats.queuedBytes-=batch.back().data.size();
                        if(!unlimited){--allowance;byteAllowance-=batch.back().data.size();}
                    }
                    if(batch.empty())continue;
                    batchDeadline=windowDeadline;batchUnlimited=unlimited;
                    active=true;
                }
                const auto start=Clock::now();
                int timer=-1;
                if(!freeTimers.empty()) {
                    timer=int(freeTimers.back());freeTimers.pop_back();
                    OTK_ERROR_CHECK(cuEventRecord(timers[timer].begin,stream));
                }
                activeMappingBudget=mappingBudget.get();
                double sdkMs=0;
                const bool captureBatch=inlineBatches()&&!batch.front().data.empty();
                if(captureBatch)beginPreparedTextureBatch(owner,stream);
                {
                    size_t executed=0;
                    for(auto& job:batch) {
                        if(executed&&!batchUnlimited&&Clock::now()>=batchDeadline)break;
                        activeJob=&job;job.upload(stream);++executed;
                    }
                    activeJob=nullptr;
                    sdkMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                    if(captureBatch)endPreparedTextureBatch();
                    // An allocation/eviction may itself exceed the window.
                    // Return untouched jobs, retaining their pending markers;
                    // never stack more SDK calls behind that overrun.
                    if(executed<batch.size()) {
                        std::lock_guard<std::mutex> lock(mutex);
                        for(size_t i=batch.size();i>executed;--i) {
                            stats.queuedBytes+=batch[i-1].data.size();jobs.push_front(std::move(batch[i-1]));
                        }
                        batch.resize(executed);
                    }
                }
                activeMappingBudget=nullptr;activeJob=nullptr;
                if(timer>=0){OTK_ERROR_CHECK(cuEventRecord(timers[timer].end,stream));pendingTimers.push_back(unsigned(timer));}
                OTK_ERROR_CHECK(cuEventRecord(phaseReady,stream));
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    stats.submitted+=batch.size();stats.submitMs+=ms;
                    stats.maxSubmitMs=std::max(stats.maxSubmitMs,ms);
                    for(const auto& job:batch) {
                        stats.bytes+=job.data.size();
                        stats.readyLatencyMs+=std::chrono::duration<double,std::milli>(start-job.ready).count();
                        pages.erase(job.page);
                    }
                    timeAllowance-=ms;active=false;
                }
                wake.notify_all();
                if(recordSlow&&ms>=20.)std::fprintf(stderr,
                    "[CLAS upload overrun] t=%.4f pages=%u SDK_ms=%.4f map_copy_ms=%.4f\n",
                    std::chrono::duration<double,std::milli>(start.time_since_epoch()).count(),unsigned(batch.size()),sdkMs,ms-sdkMs);
            }
            OTK_ERROR_CHECK(cuStreamSynchronize(stream));
            pollTimers();
            for(const auto& timer:timers){OTK_ERROR_CHECK(cuEventDestroy(timer.begin));OTK_ERROR_CHECK(cuEventDestroy(timer.end));}
            mappingBudget.reset();
            OTK_ERROR_CHECK(cuStreamDestroy(stream));
        } catch(const std::exception& error) {
            std::fprintf(stderr,"Prepared texture upload failed: %s\n",error.what());
            std::terminate();
        }
    }
public:
    explicit PreparedUploads(void* p):owner(p){
        OTK_ERROR_CHECK(cuCtxGetCurrent(&context));
        OTK_ERROR_CHECK(cuEventCreate(&phaseReady,CU_EVENT_DISABLE_TIMING));
        if(const auto* value=std::getenv("ZENO_CLAS_TEXTURE_UPLOAD_MS")) {
            const auto ms=std::strtod(value,nullptr);
            if(ms>=.1 && ms<=16.)stationaryBudget=ms;
        }
        if(const auto* value=std::getenv("ZENO_CLAS_TEXTURE_SUBMIT_WINDOW_MS"))
            submissionWindowMs=std::clamp(std::strtod(value,nullptr),0.,8.);
    }
    ~PreparedUploads(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;unlimited=true;}
        wake.notify_all();if(worker.joinable())worker.join();
        if(phaseReady)OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(phaseReady));
    }
    bool push(Job job) {
        std::lock_guard<std::mutex> lock(mutex);
        if(pages.count(job.page) || pages.size()>=1024 || stats.queuedBytes+job.data.size()>(64ull<<20))return false;
        pages.insert(job.page);stats.queuedBytes+=job.data.size();++stats.prepared;
        jobs.push_back(std::move(job));
        if(!worker.joinable())worker=std::thread([this]{run();});
        wake.notify_all();return true;
    }
    bool contains(unsigned int page) {
        std::lock_guard<std::mutex> lock(mutex);return pages.count(page)!=0;
    }
    void updatePriorities(const uint64_t* keys,size_t count) {
        std::lock_guard<std::mutex> lock(mutex);priorities.update(keys,count);
    }
    void pause() {
        std::unique_lock<std::mutex> lock(mutex);
        // A zero-gap render loop used to revoke the next submission allowance
        // before the worker could use it. Let the already granted bounded
        // window run, then quiesce before rays. CPU readers remain independent.
        if(submissionWindowMs>0&&!unlimited)
            wake.wait_until(lock,windowDeadline,[&]{return (jobs.empty()&&!active)||!allowance||timeAllowance<=0;});
        unlimited=false;allowance=0;timeAllowance=0;
        wake.notify_all();
        // Close the bounded submission window before tracing. Wait only for
        // the one active CPU driver call, then order its last copy on the GPU;
        // never wait for the entire CPU preparation queue or the IO stream.
        wake.wait(lock,[&]{return !active;});
        if(stats.submitted!=lastBarrierSubmission) {
            lastBarrierSubmission=stats.submitted;
            OTK_ERROR_CHECK(cuStreamWaitEvent(nullptr,phaseReady,0));
        }
    }
    void notify(){wake.notify_all();}
    void advance(bool interactive,bool moving,double headroomMs,bool geometryBusy,bool primaryPending) {
        std::lock_guard<std::mutex> lock(mutex);
        unlimited=!interactive;
        // Soft driver-submission budget, not a claim of hard GPU preemption.
        // A single sparse-map/driver call can exceed this interval.
        allowance=moving?(submissionWindowMs>0?16u:4u):(submissionWindowMs>0?64u:32u);
        byteAllowance=(submissionWindowMs>0&&!moving?4ull:2ull)<<20;
        timeAllowance=submissionWindowMs>0?textureServiceWindow(submissionWindowMs,headroomMs,moving,geometryBusy,primaryPending):
            std::min(moving?.5:stationaryBudget,std::max(.1,headroomMs));
        windowDeadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(timeAllowance));
        wake.notify_all();
    }
    void drain() {
        std::unique_lock<std::mutex> lock(mutex);unlimited=true;wake.notify_all();
        wake.wait(lock,[&]{return jobs.empty()&&!active;});
    }
    PreparedTextureStats snapshot() {
        std::lock_guard<std::mutex> lock(mutex);auto result=stats;
        result.queued=unsigned(pages.size());result.inFlight=pendingTextureUploads(owner);
        if(result.submitted)result.readyLatencyMs/=result.submitted;
        return result;
    }
};
std::mutex registryMutex;
std::map<void*,std::shared_ptr<PreparedUploads>> registry;
std::shared_ptr<PreparedUploads> find(void* owner) {
    std::lock_guard<std::mutex> lock(registryMutex);const auto it=registry.find(owner);
    return it==registry.end()?nullptr:it->second;
}
} // namespace
void beforeTextureMapping(CUstream stream){if(activeMappingBudget)activeMappingBudget->before(stream);}
void afterTextureMapping(CUstream stream){if(activeMappingBudget)activeMappingBudget->after(stream);}
void configurePreparedTextureUploads(void* owner) {
    auto queue=std::make_shared<PreparedUploads>(owner);
    std::lock_guard<std::mutex> lock(registryMutex);registry.emplace(owner,std::move(queue));
}
bool usePreparedTextureUploads(void* owner){return bool(find(owner));}
bool executingPreparedTextureUpload(void* owner,unsigned int page){return activeJob&&activeJob->owner==owner&&activeJob->page==page;}
unsigned int preparedTextureReclaimPages(void* owner) {
    static const bool credits=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_RECLAIM_CREDITS");return !v||*v!='0';}();
    return credits&&activeJob&&activeJob->owner==owner?textureReclaimPages(activeJob->data.size()):8u;
}
bool preparedTextureUploadPending(void* owner,unsigned int page){
    if(executingPreparedTextureUpload(owner,page))return false;
    auto p=find(owner);return p&&p->contains(page);
}
bool enqueuePreparedTextureUpload(void* owner,unsigned int page,std::vector<char> data,std::function<void(CUstream)> upload,uint64_t resourceGroup){
    auto p=find(owner);return p&&p->push({owner,page,std::move(data),std::move(upload),Clock::now(),resourceGroup});
}
bool copyPreparedTextureData(void* owner,unsigned int page,char* destination,size_t capacity){
    if(!executingPreparedTextureUpload(owner,page) || activeJob->data.empty())return false;
    if(activeJob->data.size()>capacity)throw std::runtime_error("Prepared texture exceeds its transfer buffer");
    std::memcpy(destination,activeJob->data.data(),activeJob->data.size());return true;
}
void updatePreparedTexturePriorities(void* owner,const uint64_t* keys,size_t count){if(auto p=find(owner))p->updatePriorities(keys,count);}
void advancePreparedTextureUploads(void* owner,bool interactive,bool moving,double headroomMs,bool geometryBusy,bool primaryPending){if(auto p=find(owner))p->advance(interactive,moving,headroomMs,geometryBusy,primaryPending);}
void pausePreparedTextureUploads(void* owner){if(auto p=find(owner))p->pause();}
void wakePreparedTextureUploads(void* owner){
    // Do not transfer final queue ownership to the completion thread: its
    // destructor joins the submitter, which can itself enqueue completions.
    std::lock_guard<std::mutex> lock(registryMutex);const auto it=registry.find(owner);
    if(it!=registry.end())it->second->notify();
}
void drainPreparedTextureUploads(void* owner){if(auto p=find(owner))p->drain();}
void releasePreparedTextureUploads(void* owner){
    std::shared_ptr<PreparedUploads> p;
    {std::lock_guard<std::mutex> lock(registryMutex);auto it=registry.find(owner);if(it==registry.end())return;p=std::move(it->second);registry.erase(it);}
    p.reset();
}
PreparedTextureStats preparedTextureStats(void* owner){if(auto p=find(owner))return p->snapshot();return {};}
}} // namespace zeno::cluster
