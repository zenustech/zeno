#pragma once

#include <zeno/zeno.h>
#include <rapidjson/document.h>

namespace zeno {

zany parseObjectFromUi(rapidjson::Value const &x);

}
