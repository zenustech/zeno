#pragma once

#include <functional>
#include <string_view>
#include "vec.h"

namespace fdb {

void write_dense_vdb
    ( std::string_view path
    , std::function<float(vec3I)> sampler
    , vec3I size
    );

void write_dense_vdb
    ( std::string_view path
    , std::function<vec3f(vec3I)> sampler
    , vec3I size
    );

}
