#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace zeno::cluster {
// Safe under the runtime two-sided opaque policy. Alpha modes always retain
// any-hit, as do authored one-sided transmissive surfaces. The ray restores
// enforced any-hit when that runtime policy is disabled.
inline bool opaqueShadowCandidate(bool alpha,bool doubleSided,float transmission) {
    return !alpha && (doubleSided || transmission<=0.f);
}
struct OpaqueSlotPolicy {
    std::vector<std::vector<uint8_t>> slots;
    std::vector<bool> seen;
    explicit OpaqueSlotPolicy(std::size_t geometries=0):slots(geometries),seen(geometries,false){}
    void add(uint32_t geometry,const std::vector<uint8_t>& placement) {
        auto& result=slots.at(geometry);
        if(!seen.at(geometry)){result=placement;seen[geometry]=true;return;}
        // CLAS are shared by instances. A slot is safe only if every instance
        // using that geometry agrees. Unknown/missing slots remain any-hit.
        for(std::size_t i=0;i<result.size();++i)result[i]=result[i] && i<placement.size() && placement[i];
    }
    bool opaque(uint32_t geometry,uint32_t slot) const {
        return geometry<slots.size() && slot<slots[geometry].size() && slots[geometry][slot];
    }
};
}
