#pragma once

#include "common.h"
#include <tuple>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::string>
    > compile_to_assembly
    ( std::string const &code
    );
}
