#pragma once

#include "common.h"
#include <tuple>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::string>
    > zfx_compile
    ( std::string const &code
    );
}
