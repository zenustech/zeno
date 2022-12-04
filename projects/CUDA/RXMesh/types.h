#pragma once
#include <stdint.h>

namespace zeno::rxmesh {

using locationT = uint32_t;
enum : locationT {
    LOCATION_NONE = 0x00,
    HOST          = 0x01,
    DEVICE        = 0x02,
    LOCATION_ALL  = 0x0F,
};

using layoutT = uint32_t;
enum : layoutT {
    AoS = 0x00,
    SoA = 0x01,
};

enum class Op {
    VV = 0,
    VE = 1,
    VF = 2,
    FV = 3,
    FE = 4,
    FF = 5,
    EV = 6,
    EE = 7,
    EF = 8,
};
}  // namespace rxmesh