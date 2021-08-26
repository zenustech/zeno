#pragma once

#include <functional>
#include <string>
#include "vec.h"

namespace fdb {

void write_dense_vdb
    ( std::string path
    , std::function<float(vec3i)> sampler
    , vec3i start
    , vec3i stop
    );

void write_dense_vdb
    ( std::string path
    , std::function<vec3f(vec3i)> sampler
    , vec3i start
    , vec3i stop
    );

}
