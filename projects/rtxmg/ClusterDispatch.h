#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace zeno::cluster {
inline uint32_t checkedWorkItems(uint64_t preceding,uint32_t residents,uint32_t placements) {
    constexpr uint64_t limit=std::numeric_limits<uint32_t>::max()/32;
    const uint64_t count=uint64_t(residents)*placements;
    if(preceding>limit||count>limit-preceding)throw std::runtime_error("CLAS resident dispatch capacity exceeded");
    return static_cast<uint32_t>(count);
}
}
