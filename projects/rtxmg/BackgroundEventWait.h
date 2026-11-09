#pragma once
#include <cuda.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace zeno::cluster {
// A captured background cudaEventSynchronize held a Windows CUDA driver lock
// needed by optixLaunch. Leave the driver between queries while retaining the
// exact completion fence required before publishing or reusing GPU data.
inline bool backgroundEventQueriesEnabled() {
    static const bool enabled=[] {
        const auto* value=std::getenv("ZENO_CLAS_BACKGROUND_EVENT_QUERY");
        if(value)return std::strcmp(value,"0")!=0;
#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }();
    return enabled;
}
inline void pauseBackgroundEventQuery() {
#ifdef _WIN32
    // A private high-resolution timer avoids both a hot driver-query loop
    // and Sleep(1)'s coarse timer rounding. No global timer resolution change.
    struct Timer {
        enum : DWORD { highResolution=0x2u }; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
        HANDLE handle=CreateWaitableTimerExW(nullptr,nullptr,highResolution,TIMER_MODIFY_STATE|SYNCHRONIZE);
        ~Timer(){if(handle)CloseHandle(handle);}
    };
    static thread_local Timer timer;
    LARGE_INTEGER due;due.QuadPart=-1000; // 100 microseconds, in 100 ns units.
    if(timer.handle&&SetWaitableTimer(timer.handle,&due,0,nullptr,nullptr,FALSE)) {
        if(WaitForSingleObject(timer.handle,INFINITE)==WAIT_OBJECT_0)return;
    }
    // Old Windows versions may not support high-resolution waitable timers.
    // A coarse sleep is preferable to a hot query loop on that fallback path.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#else
    std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
}
inline CUresult waitBackgroundEvent(CUevent event) {
    if(!backgroundEventQueriesEnabled())return cuEventSynchronize(event);
    for(;;) {
        const auto status=cuEventQuery(event);
        if(status!=CUDA_ERROR_NOT_READY)return status;
        pauseBackgroundEventQuery();
    }
}
}
