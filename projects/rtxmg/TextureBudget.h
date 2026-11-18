#pragma once
#include <algorithm>
#include <cstdint>

namespace zeno::cluster {
// Limits sparse backing, not image quality. Actual free memory already includes
// live geometry/CLAS, AS banks, render targets and other processes: do not charge
// those allocations a second time. Reserve covers future build/scratch growth.
struct TextureBudgetPolicy {
    uint64_t ceiling=0,quantum=0,limit=0;
    unsigned int hotSamples=0,coolSamples=0;
    uint64_t floor() const {return std::min(ceiling,std::max(quantum,uint64_t(256)<<20));}
    uint64_t reserve(uint64_t total,uint64_t buildGrowth) const {
        return std::max(uint64_t(512)<<20,total/20)+std::min(buildGrowth,uint64_t(2)<<30);
    }
    uint64_t initial(uint64_t free,uint64_t total) {
        const auto keep=reserve(total,0);
        limit=std::clamp(free>keep?(free-keep)/quantum*quantum:floor(),floor(),ceiling);
        return limit;
    }
    // One decision per fresh memory sample, never one per rendered frame.
    uint64_t update(uint64_t free,uint64_t total,uint64_t buildGrowth,bool idle) {
        const auto keep=reserve(total,buildGrowth);
        if(free<keep){++hotSamples;coolSamples=0;}
        else if(free>keep+(uint64_t(256)<<20)){++coolSamples;hotSamples=0;}
        else hotSamples=coolSamples=0;
        // SDK shrink invalidates mappings synchronously. Require two pressure
        // samples and a quiescent stationary renderer, then remove one arena.
        if(hotSamples>=2 && idle && limit>floor()) {hotSamples=0;limit=std::max(floor(),limit-quantum);}
        // Growth changes only the lazy allocator ceiling. No eager VRAM grab.
        if(coolSamples>=3 && limit<ceiling) {
            coolSamples=0;
            const auto spare=(free-keep)/2/quantum*quantum;
            const auto step=std::max(quantum,std::min(uint64_t(512)<<20,spare));
            limit=std::min(ceiling,limit+step);
        }
        return limit;
    }
};
}
