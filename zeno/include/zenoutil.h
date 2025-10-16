#pragma once

#include <zeno/utils/api.h>
#include <string>

namespace zeno {
    ZENO_API bool runPython(const std::string& script);
}