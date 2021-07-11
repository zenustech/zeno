#pragma once

#include "common.h"
#include "x64/Executable.h"

struct Program {
    std::unique_ptr<ExecutableInstance> executable;

    std::vector<float> consts;
    std::vector<float> locals;
    std::vector<float *> chptrs;

    void set_local_mem_size(size_t size) {
        locals.resize(size);
    }

    void set_channel_pointer(int i, float *ptr) {
        if (chptrs.size() < i + 1) {
            chptrs.resize(i + 1);
        }
        chptrs[i] = ptr;
    }

    void operator()() {
        auto rbx = locals.data();
        auto rcx = consts.data();
        auto rdx = chptrs.data();
        (*executable)(0, (uintptr_t)rbx, (uintptr_t)rcx, (uintptr_t)rdx);
    }
};

std::unique_ptr<Program> assemble_program(std::string const &lines);
