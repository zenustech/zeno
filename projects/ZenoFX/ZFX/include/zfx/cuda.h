#pragma once

#include <memory>
#include <cstring>
#include <string>
#include <map>

namespace zfx::cuda {

struct Executable {
    std::string code;

    auto const &get_cuda_source() {
        return code;
    }
};

struct Assembler {
    std::map<std::string, std::string> cache;

    static std::string impl_assemble
        ( std::string const &lines
        );

    std::string assemble(std::string const &lines) {
        if (auto it = cache.find(lines); it != cache.end()) {
            return it->second;
        }
        auto code = impl_assemble(lines);
        cache[lines] = code;
        return code;
    }
};

}
