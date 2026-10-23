#pragma once
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

namespace zeno::cluster {
// Opt-in CPU ranges correlate frame stages with CUDA API waits in Nsight
// Systems. In normal runs no NVTX calls are made.
class TimelineRange {
    bool enabled;
public:
    explicit TimelineRange(const char* name) {
        static const bool active=[] { const auto* v=std::getenv("ZENO_OPTIX_TIMELINE");return v&&v[0]=='1'; }();
        enabled=active;
        if(enabled)nvtxRangePushA(name);
    }
    ~TimelineRange(){if(enabled)nvtxRangePop();}
    TimelineRange(const TimelineRange&)=delete;
};
}
