#pragma once

#include "common.h"
#include "cpu/Executable.h"
#include <map>

struct Program {
    std::unique_ptr<ExecutableInstance> executable;
    std::map<std::string, int> symtable;

    std::vector<float> consts;
    std::vector<float *> chptrs;

    void set_constant(int i, float value) {
        if (consts.size() < i + 1) {
            consts.resize(i + 1);
        }
        consts[i] = value;
    }

    void set_channel_pointer(int i, float *ptr) {
        if (chptrs.size() < i + 1) {
            chptrs.resize(i + 1);
        }
        chptrs[i] = ptr;
    }

    void operator()() {
        auto rcx = consts.data();
        auto rdx = chptrs.data();
        (*executable)(0, (uintptr_t)rcx, (uintptr_t)rdx);
    }
};

std::unique_ptr<Program> assemble_program(std::string const &lines);
