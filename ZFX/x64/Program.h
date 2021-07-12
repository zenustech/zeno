#pragma once

#include "common.h"
#include "x64/Executable.h"

namespace zfx::x64 {

struct Program {
    std::unique_ptr<ExecutableInstance> executable;

    std::vector<float> consts;
    std::vector<std::array<float, 4>> locals;
    std::vector<float *> chptrs;

    float *&channel_pointer(int i) {
        if (chptrs.size() < i + 1) {
            chptrs.resize(i + 1);
        }
        return chptrs[i];
    }

    void execute() {
        auto rbx = locals.data();
        auto rcx = consts.data();
        auto rdx = chptrs.data();
        (*executable)(0, (uintptr_t)rbx, (uintptr_t)rcx, (uintptr_t)rdx);
    }

    static std::unique_ptr<Program> assemble(std::string const &lines);
};

}
