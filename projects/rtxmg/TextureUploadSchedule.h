#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace zeno::cluster {
inline unsigned int textureReclaimPages(size_t bytes) {
    return static_cast<unsigned int>(std::max<size_t>(1,bytes/65536+(bytes%65536!=0)));
}
// Fixed-size hints from the already-ranked GPU demand list. No unbounded
// history, new IO or cancellation of SDK-owned requests.
struct TextureUploadPriorities {
    static constexpr size_t Capacity=128;
    struct Hint {uint32_t page;uint64_t score;};
    std::array<Hint,Capacity> hints{};
    size_t count=0;
    void update(const uint64_t* keys,size_t size) {
        count=0;
        for(size_t i=0;i<std::min(size,Capacity);++i)if(keys[i])
            hints[count++]={0xffffffffu-uint32_t(keys[i]),keys[i]};
        std::sort(hints.begin(),hints.begin()+count,[](auto a,auto b){return a.page<b.page;});
    }
    uint64_t score(uint32_t page,bool bootstrap) const {
        if(bootstrap)return ~uint64_t(0);
        const auto end=hints.begin()+count;
        const auto it=std::lower_bound(hints.begin(),end,page,[](auto h,uint32_t p){return h.page<p;});
        return it!=end&&it->page==page?it->score:0;
    }
};
inline double textureServiceWindow(double ceiling,double headroom,bool moving,bool geometryBusy,bool primaryPending) {
    // Preserve stationary primary delivery throughput. Reducing this window
    // to render headroom alone starves cold views while geometry converges.
    // Once primary detail is ready, background work yields to render headroom.
    const double shared=std::clamp(headroom,.5,2.);
    return std::min(ceiling,moving?2.:primaryPending?8.:geometryBusy?std::min(1.,shared):shared);
}
}
