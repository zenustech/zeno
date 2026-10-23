#pragma once
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <cuda.h>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdio>

namespace demandLoading {
// Paging callbacks take SDK locks also held by threads making CUDA calls.
// Running them inside cuLaunchHostFunc can deadlock the Windows driver. An
// event marks the request readback; an ordinary CPU thread handles the batch.
// The caller must drain its OTK tickets before destroying the demand loader.
class CudaCallback {
    struct Job {
        CUcontext context{};
        CUevent ready{};
        std::unique_ptr<CudaCallback> callback;
        ~Job(){if(ready)OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(ready));}
    };
    class Dispatcher {
        std::mutex mutex;
        std::condition_variable wake;
        std::deque<std::unique_ptr<Job>> pending;
        bool stopping=false;
        bool draining=false;
        bool active=false;
        std::thread worker;
        void run() noexcept {
            try {
                for(;;) {
                    std::unique_ptr<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        wake.wait(lock,[&]{return stopping||!pending.empty();});
                        if(pending.empty())return;
                        job=std::move(pending.front());pending.pop_front();
                        active=true;
                    }
                    OTK_ERROR_CHECK(cuCtxSetCurrent(job->context));
                    OTK_ERROR_CHECK(cuEventSynchronize(job->ready));
                    job->callback->callback();
                    job.reset();
                    {std::lock_guard<std::mutex> lock(mutex);active=false;}
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
        Dispatcher()=default;
        ~Dispatcher(){drain();}
        void push(std::unique_ptr<Job> job){
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock,[&]{return !draining;});
                pending.push_back(std::move(job));
                if(!worker.joinable())worker=std::thread([this]{run();});
            }
            wake.notify_all();
        }
        void drain(){
            std::unique_lock<std::mutex> lock(mutex);
            wake.wait(lock,[&]{return !draining;});draining=true;
            wake.wait(lock,[&]{return pending.empty()&&!active;});
            stopping=true;lock.unlock();wake.notify_all();
            if(worker.joinable())worker.join();
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
        OTK_ERROR_CHECK(cuEventCreate(&job->ready,CU_EVENT_DISABLE_TIMING));
        OTK_ERROR_CHECK(cuEventRecord(job->ready,stream));
        dispatcher().push(std::move(job));
    }
    static void drain(){dispatcher().drain();}
};
}
