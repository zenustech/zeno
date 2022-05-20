#pragma once

#include <zeno/zeno.h>
#include <rapidjson/document.h>

using namespace rapidjson;

namespace zeno {
    zany parseObject(Value const &x);
}