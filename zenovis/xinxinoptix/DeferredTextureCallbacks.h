#pragma once
#include "CompletedTextureUpload.h"
#include "BackgroundEventWait.h"
#include "TextureSyncEvent.h"
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <cuda.h>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace demandLoading {
// Paging callbacks take SDK locks also held by threads making CUDA calls.
// Running them inside cuLaunchHostFunc can deadlock the Windows driver. An
// event marks the request readback; an ordinary CPU thread handles the batch.
// The caller must drain its OTK tickets before destroying the demand loader.
class CudaCallback {
    using Clock=std::chrono::steady_clock;
public:
    using Stats=zeno::cluster::TextureCallbackStats;
    struct EventStats {uint64_t created,reused,cached;};
private:
    struct Job {
        CUcontext context{};
        CUevent ready{};
        std::unique_ptr<CudaCallback> callback;
        Clock::time_point submitted=Clock::now();
        ~Job(){if(ready) {
            const auto status=zeno::cluster::setTextureContext(context);
            if(status==CUDA_SUCCESS)OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(ready));
            else OTK_ERROR_CHECK_NOTHROW(status);
        }}
    };
    class Dispatcher {
        struct CachedEvent {CUcontext context;CUevent event;};
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<std::unique_ptr<Job>> pending;
        bool stopping=false;
        bool draining=false;
        bool active=false;
        Clock::time_point activeSince{};
        uint64_t peak=0,completed=0;
        uint64_t eventsCreated=0,eventsReused=0;
        std::vector<CachedEvent> events;
        std::thread worker;
        static bool reuseEvents() {
            static const bool enabled=[] {const auto* v=std::getenv("ZENO_CLAS_TEXTURE_CALLBACK_EVENT_REUSE");return !v||*v!='0';}();
            return enabled;
        }
        void run() noexcept {
            try {
                for(;;) {
                    std::unique_ptr<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        wake.wait(lock,[&]{return stopping||!pending.empty();});
                        if(pending.empty()) {
                            lock.unlock();
                            // Flush while the callback thread is still running.
                            // Joining a writer from DLL TLS detach deadlocks on
                            // the Windows loader lock during thread shutdown.
                            zeno::cluster::releaseTexturePagingRecorder();
                            return;
                        }
                        job=std::move(pending.front());pending.pop_front();
                        active=true;
                        activeSince=job->submitted;
                    }
                    OTK_ERROR_CHECK(zeno::cluster::setTextureContext(job->context));
                    OTK_ERROR_CHECK(zeno::cluster::waitBackgroundEvent(job->ready));
                    job->callback->callback();
                    job->callback.reset();
                    if(reuseEvents()) {
                        // The readback event is complete and only this job
                        // owns its handle. No producer can reuse it earlier.
                        std::lock_guard<std::mutex> lock(mutex);
                        if(events.size()<64){events.push_back({job->context,job->ready});job->ready=nullptr;}
                    }
                    job.reset();
                    {std::lock_guard<std::mutex> lock(mutex);active=false;++completed;}
                    wake.notify_all();
                }
            } catch(const std::exception& error) {
                // A failed CUDA completion cannot produce valid page mappings.
                // Fail explicitly rather than leaving tickets waiting forever.
                std::fprintf(stderr,"OTK request dispatch failed: %s\n",error.what());
                std::terminate();
            }
        }
    public:
        Dispatcher(){events.reserve(64);}
        ~Dispatcher(){drain();}
        CUevent acquire(CUcontext context) {
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait(lock,[&]{return !draining;});
            if(reuseEvents())for(auto it=events.begin();it!=events.end();++it)if(it->context==context) {
                const auto event=it->event;events.erase(it);++eventsReused;return event;
            }
            lock.unlock();CUevent event{};
            OTK_ERROR_CHECK(cuEventCreate(&event,CU_EVENT_DISABLE_TIMING));
            lock.lock();++eventsCreated;return event;
        }
        void push(std::unique_ptr<Job> job){
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock,[&]{return !draining;});
                pending.push_back(std::move(job));
                peak=std::max<uint64_t>(peak,pending.size()+unsigned(active));
                if(!worker.joinable())worker=std::thread([this]{run();});
            }
            wake.notify_all();
        }
        Stats stats(){
            std::lock_guard<std::mutex> lock(mutex);
            const auto now=Clock::now();
            const auto age=[&](auto t){return std::chrono::duration<double,std::milli>(now-t).count();};
            return {pending.size(),unsigned(active),peak,completed,
                pending.empty()?0.:age(pending.front()->submitted),active?age(activeSince):0.};
        }
        EventStats eventStats(){std::lock_guard<std::mutex> lock(mutex);return {eventsCreated,eventsReused,events.size()};}
        void drain(){
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait(lock,[&]{return !draining;});draining=true;
            wake.wait(lock,[&]{return pending.empty()&&!active;});
            stopping=true;lock.unlock();wake.notify_all();
            if(worker.joinable())worker.join();
            lock.lock();auto retired=std::move(events);events.clear();
            const EventStats totals{eventsCreated,eventsReused,retired.size()};lock.unlock();
            // Drain while every producing CUDA context is still alive.
            for(const auto& event:retired) {
                const auto status=cuCtxPushCurrent(event.context);
                if(status==CUDA_SUCCESS) {
                    OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(event.event));
                    CUcontext ignored{};OTK_ERROR_CHECK_NOTHROW(cuCtxPopCurrent(&ignored));
                } else OTK_ERROR_CHECK_NOTHROW(status);
            }
            if(!retired.empty()&&std::getenv("ZENO_OPTIX_CAMERA_RECORD"))std::fprintf(stderr,
                "[CLAS callback events] created=%llu reused=%llu retired=%llu\n",
                static_cast<unsigned long long>(totals.created),static_cast<unsigned long long>(totals.reused),
                static_cast<unsigned long long>(totals.cached));
            lock.lock();stopping=false;draining=false;lock.unlock();wake.notify_all();
        }
    };
    static Dispatcher& dispatcher(){static Dispatcher instance;return instance;}
public:
    virtual ~CudaCallback()=default;
    virtual void callback()=0;
    static void enqueue(CUstream stream,CudaCallback* callback){
        auto job=std::make_unique<Job>();job->callback.reset(callback);
        OTK_ERROR_CHECK(cuCtxGetCurrent(&job->context));
        job->ready=dispatcher().acquire(job->context);
        OTK_ERROR_CHECK(cuEventRecord(job->ready,stream));
        dispatcher().push(std::move(job));
    }
    static void drain(){dispatcher().drain();}
    static Stats stats(){return dispatcher().stats();}
    static EventStats eventStats(){return dispatcher().eventStats();}
};
}
