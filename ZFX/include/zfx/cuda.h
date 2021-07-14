#pragma once

#include <memory>
#include <cstring>
#include <string>
#include <map>

namespace zfx::cuda {

struct Program {
    std::string code;

    static std::unique_ptr<Program> assemble
        ( std::string const &lines
        , std::map<int, std::string> const &consts
        );
};

}
