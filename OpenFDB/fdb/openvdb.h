#pragma once

#include <functional>
#include <string_view>
#include "types.h"

namespace fdb {

void write_dense_vdb
    ( std::string_view path
    , std::function<Qfloat(Quint3)> sampler
    , Quint3 size
    );

void write_dense_vdb
    ( std::string_view path
    , std::function<Qfloat3(Quint3)> sampler
    , Quint3 size
    );

}
