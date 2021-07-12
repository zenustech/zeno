#pragma once

#include "common.h"
#include <cstring>

namespace zfx::x64 {

struct Program {
    float consts[4096];
    uint8_t *mem = nullptr;
    size_t memsize;

    struct Context {
        Program *prog;
        float buffer[8192];

        Context(Program *prog_ = nullptr) {
            prog = prog_;
            memset(buffer, 0, sizeof(buffer));
        }

        void execute() {
            auto entry = (void (*)())prog->mem;
            asm volatile (
                "call *%1"
                :
                : "" (entry)
                , "c" ((uintptr_t)(void *)prog->consts)
                , "d" ((uintptr_t)(void *)buffer)
                : "cc", "memory"
                );
        }

        float *&channel_pointer(int chid) {
            return chid[(float **)(char *)buffer];
        }
    };

    inline Context make_context() {
        return {this};
    }

    static std::unique_ptr<Program> assemble(std::string const &lines);
};

}
