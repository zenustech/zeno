//
// Created by admin on 2022/5/7.
//

#pragma once

#include <map>
#include <memory>
namespace zfx::x64 {
    struct Executable {

    };

    struct Assembler {
        std::map<std::string, std::unique_ptr<Executable>> cache;

    };
}
