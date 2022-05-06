#pragma once

#include <string_view>
#include <functional>
#include "vec.h"

namespace zinc {

void writevdb
    ( std::string_view path
    , std::function<float(vec3I)> sampler
    , vec3I size
    );

void writevdb
    ( std::string_view path
    , std::function<vec3f(vec3I)> sampler
    , vec3I size
    );

}
