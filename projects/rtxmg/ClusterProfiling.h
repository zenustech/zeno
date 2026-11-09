#pragma once
#include <cstdlib>
#include <cstdint>
#include <nvtx3/nvToolsExt.h>

namespace zeno::cluster {
// Opt-in CPU ranges correlate frame stages with CUDA API waits in Nsight
// Systems. In normal runs no NVTX calls are made.
class TimelineRange {
    bool enabled;
public:
    explicit TimelineRange(const char* name,uint64_t payload=0) {
        static const bool active=[] { const auto* v=std::getenv("ZENO_OPTIX_TIMELINE");return v&&v[0]=='1'; }();
        enabled=active;
        if(enabled) {
            nvtxEventAttributes_t event{};event.version=NVTX_VERSION;event.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType=NVTX_MESSAGE_TYPE_ASCII;event.message.ascii=name;
            event.payloadType=NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;event.payload.ullValue=payload;
            nvtxRangePushEx(&event);
        }
    }
    ~TimelineRange(){if(enabled)nvtxRangePop();}
    TimelineRange(const TimelineRange&)=delete;
};
}
