#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace zeno::nvrtc_worker {

struct CompileResult
{
    bool success = false;
    std::string data;
    std::string log;
};

CompileResult compile(
    const char* source,
    const char* macro,
    const char* name,
    const std::vector<const char*>& compilerOptions);

} // namespace zeno::nvrtc_worker
