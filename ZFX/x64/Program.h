#pragma once

#include "common.h"

namespace zfx::x64 {

struct Program {
    float consts[4096];
    uint8_t *mem = nullptr;
    size_t memsize;

    struct Context {
        Program *prog;
        float buffer[8192];

        void execute() {
            auto entry = (void (*)())prog->mem;
            asm volatile (
                "call *%1"
                :
                : "c" ((uintptr_t)(void *)prog->consts)
                , "d" ((uintptr_t)(void *)buffer)
                , "" (entry)
                : "cc", "memory"
                );
        }

        float *&channel_pointer(int chid) {
            return chid[(float **)(char *)buffer];
        }
    };

    Context make_context() {
        return {this};
    }

    static std::unique_ptr<Program> assemble(std::string const &lines);
};

}
