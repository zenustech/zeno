#ifndef __CUDA_MEM_TRACER_HPP__
#define __CUDA_MEM_TRACER_HPP__

#include <string>
#include <array>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

struct CudaMemInfo
{
    std::array<void *, 4> callStack;
    std::string callee;
    std::string caller;
    std::string file;
    void *pointer{nullptr};
    std::size_t size{0};
    unsigned int line;
};

class CudaMemTracer
{
private:
    static constexpr char recordDir[] = "./cudaMemRecord";
    std::unordered_map<void *, CudaMemInfo> memTable_;
    bool isUpdateMemTable_ = false;
    std::mutex mutex_;

    CudaMemTracer()
    {
        std::thread recordThread([&]()
        {
            while (true)
            {
                std::vector<CudaMemInfo> infoArray(0);
                {
                    std::lock_guard<std::mutex> lockGuard(mutex_);
                    if (isUpdateMemTable_)
                    {
                        isUpdateMemTable_ = false;
                        infoArray.resize(memTable_.size());
                        int i = 0;
                        for (const auto &[_, memInfo] : memTable_)
                        {
                            infoArray[i] = memInfo;
                            ++i;
                        }
                    }
                }

                if (infoArray.size() != 0)
                {
                    std::sort(
                        infoArray.begin(),
                        infoArray.end(),
                        [](const auto &lhs, const auto &rhs)
                        {
                            return lhs.size >= rhs.size;
                        });
                    
                    std::cout << "===================\n";
                    std::cout << "memCount: " << infoArray.size() << "\n";
                    std::size_t sum = 0;
                    for (const auto &memInfo : infoArray)
                    {
                        std::cout << "ptr: " << memInfo.pointer << ", size: " << memInfo.size << "\n";
                        sum += memInfo.size;
                    }
                    std::cout << "memSum: " << sum << "\n";
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });
        recordThread.detach();
    }

    ~CudaMemTracer() = default;

public:
    static CudaMemTracer &getInstance()
    {
        static CudaMemTracer instance_;
        return instance_;
    }

    void pushMemInfo(
        void *ptr,
        std::size_t size,
        const std::string &callee,
        const std::string &caller,
        const std::string &file,
        const unsigned int line,
        void *callLevel0, 
        void *callLevel1, 
        void *callLevel2, 
        void *callLevel3)
    {
        std::lock_guard<std::mutex> lockGuard(mutex_);
        isUpdateMemTable_ = true;
        auto &memInfo = memTable_[ptr];
        memInfo.pointer = ptr;
        memInfo.size = size;
        memInfo.callee = callee;
        memInfo.caller = caller;
        memInfo.file = file;
        memInfo.line = line;
        memInfo.callStack[0] = callLevel0;
        memInfo.callStack[1] = callLevel1;
        memInfo.callStack[2] = callLevel2;
        memInfo.callStack[3] = callLevel3;
    }

    void popMemInfo(void *ptr)
    {
        std::lock_guard<std::mutex> lockGuard(mutex_);
        isUpdateMemTable_ = true;
        memTable_.erase(ptr);
    }

    CudaMemTracer(const CudaMemTracer &) = delete;
    CudaMemTracer(CudaMemTracer &&) = delete;
    CudaMemTracer &operator=(const CudaMemInfo &) = delete;
    CudaMemTracer &operator=(CudaMemInfo &&) = delete;
};

#endif
