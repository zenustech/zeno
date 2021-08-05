#pragma once

#include "common.h"
#include <cstdint>


namespace zeno::v2::container {

using any = std::variant
        < std::any
        , int
        , float
        >;

}
