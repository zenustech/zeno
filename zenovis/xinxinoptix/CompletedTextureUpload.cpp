#include "CompletedTextureUpload.h"
#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace zeno { namespace cluster { namespace {
struct Upload {
    CUcontext context{};
    CUevent completed{};
    void* owner{};
    unsigned int page{};
    std::function<void()> publish;
    ~Upload() { if(completed) OTK_ERROR_CHECK_NOTHROW(cuEventDestroy(completed)); }
};
class Completions {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::unique_ptr<Upload>> queue;
    std::map<void*,std::set<unsigned int>> pending;
    std::thread worker;
    bool stopping=false,draining=false;
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
                OTK_ERROR_CHECK(cuCtxSetCurrent(job->context));
                // Only this completion worker waits. IO workers can decode
                // and queue subsequent uploads while rendering continues.
                OTK_ERROR_CHECK(cuEventSynchronize(job->completed));
                job->publish();
                const auto owner=job->owner;const auto page=job->page;
                job.reset();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto it=pending.find(owner);
                    it->second.erase(page);
                    if(it->second.empty())pending.erase(it);
                }
                wake.notify_all();
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
} // namespace
void enqueueTextureUpload(CUstream stream,void* owner,unsigned int page,std::function<void()> publish) {
    auto job=std::unique_ptr<Upload>(new Upload);
    job->owner=owner;job->page=page;job->publish=std::move(publish);
    OTK_ERROR_CHECK(cuCtxGetCurrent(&job->context));
    OTK_ERROR_CHECK(cuEventCreate(&job->completed,CU_EVENT_DISABLE_TIMING));
    OTK_ERROR_CHECK(cuEventRecord(job->completed,stream));
    completions().enqueue(std::move(job));
}
bool textureUploadPending(void* owner,unsigned int page) { return completions().contains(owner,page); }
unsigned int pendingTextureUploads(void* owner) { return completions().count(owner); }
void drainTextureUploads(void* owner) { completions().drain(owner); }
}} // namespace zeno::cluster
