#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <Windows.h>
#else
#include <time.h>
#endif

namespace zeno::framecapture {
inline bool enabled() {
    static const bool value=[] {const auto* v=std::getenv("ZENO_FRAME_CAPTURE");return v&&*v;}();
    return value;
}
inline double nowMs() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double threadMs() {
#ifdef _WIN32
    FILETIME created,exited,kernel,user;
    if(!GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user))return 0;
    return (double((uint64_t(kernel.dwHighDateTime)<<32)|kernel.dwLowDateTime)+
            double((uint64_t(user.dwHighDateTime)<<32)|user.dwLowDateTime))/10000.;
#else
    timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1000.+t.tv_nsec/1.e6;
#endif
}
// Opt-in diagnostics only. The producer copies numeric rows into a bounded
// history. File formatting and IO run on a separate thread; a full writer
// queue drops a batch instead of blocking rendering. Timestamps use the same
// monotonic clock in the UI and renderer. Qt paint completion is not scanout.
class Recorder {
public:
    struct Row {uint64_t sequence{};double timestamp{},duration{};std::array<double,96> values{};};
private:
    static constexpr size_t History=128,Post=32,Queue=8;
    std::array<Row,History> history{};
    size_t head=0,count=0,post=0;
    uint64_t sequence=0,dropped=0;
    double threshold=60;
    bool all=false,stopping=false;
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<Row>> pending;
    std::thread writer;
    std::atomic<bool> failed{false};
    std::string failure;
    void submit(std::vector<Row> rows) {
        std::lock_guard<std::mutex> lock(mutex);
        if(pending.size()==Queue){++dropped;return;}
        pending.push_back(std::move(rows));ready.notify_one();
    }
    void snapshot() {
        std::vector<Row> rows;rows.reserve(count);
        for(size_t i=0;i<count;++i)rows.push_back(history[(head+History-count+i)%History]);
        submit(std::move(rows));count=0;
    }
public:
    Recorder(const char* name,const char* columns,bool allEvents=false)
        :Recorder(enabled()?std::filesystem::u8path(std::getenv("ZENO_FRAME_CAPTURE")):std::filesystem::path{},name,columns,allEvents) {}
    // Explicit destinations do not enable the renderer's GPU diagnostics.
    Recorder(const std::filesystem::path& root,const char* name,const char* columns,bool allEvents) {
        if(root.empty())return;
        all=allEvents;
        if(const auto* v=std::getenv("ZENO_FRAME_CAPTURE_ALL"))all=all||*v=='1';
        if(const auto* v=std::getenv("ZENO_FRAME_CAPTURE_MS")) {
            const double t=std::strtod(v,nullptr);if(t>0)threshold=t;
        }
        const auto file=root/(std::string(name)+"-"+std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()))+".csv");
        writer=std::thread([this,file,header=std::string(columns)] {
            try {
                std::filesystem::create_directories(file.parent_path());
                std::ofstream output(file);if(!output)throw std::runtime_error("Cannot open capture file: "+file.u8string());
                output<<"sequence,timestamp_ms,duration_ms,"<<header<<'\n'<<std::setprecision(17);
                const size_t fields=1+size_t(std::count(header.begin(),header.end(),','));
                for(;;) {
                    std::vector<Row> rows;
                    {std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stopping||!pending.empty();});
                        if(pending.empty())break;rows=std::move(pending.front());pending.pop_front();}
                    for(const auto& r:rows) {
                        output<<r.sequence<<','<<r.timestamp<<','<<r.duration;
                        for(size_t i=0;i<fields&&i<r.values.size();++i)output<<','<<r.values[i];
                        output<<'\n';
                    }
                    output.flush();
                    if(!output)throw std::runtime_error("Cannot write capture file: "+file.u8string());
                }
                output<<"# dropped_batches="<<dropped<<'\n';
                output.flush();
                if(!output)throw std::runtime_error("Cannot finish capture file: "+file.u8string());
            }catch(const std::exception& e){std::lock_guard<std::mutex> lock(mutex);failure=e.what();failed=true;}
            catch(...){std::lock_guard<std::mutex> lock(mutex);failure="Capture writer failed";failed=true;}
        });
    }
    ~Recorder() {finish();}
    void finish() {
        if(!writer.joinable())return;
        if(all||post)snapshot();
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;ready.notify_one();}
        writer.join();
    }
    uint64_t droppedBatches() { std::lock_guard<std::mutex> lock(mutex);return dropped; }
    std::string error() {std::lock_guard<std::mutex> lock(mutex);return failure;}
    void record(double start,double duration,std::initializer_list<double> values) {record(start,duration,values.begin(),values.size());}
    void record(double start,double duration,const double* values,size_t size) {
        if(!writer.joinable()||failed.load())return;
        Row r;r.sequence=++sequence;r.timestamp=start;r.duration=duration;
        std::copy_n(values,std::min(size,r.values.size()),r.values.begin());
        history[head]=r;head=(head+1)%History;count=std::min(count+1,History);
        if(all){if(count==History)snapshot();return;}
        if(duration>=threshold&&!post){snapshot();post=Post;}
        else if(post&&!--post)snapshot();
    }
};
}
