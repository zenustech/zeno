#pragma once

#include <memory>
#include <cstring>

namespace zfx::x64 {

struct Program {
    float consts[4096];
    uint8_t *mem = nullptr;
    size_t memsize;

    struct Context {
        Program *prog;
        float locals[4 * 128];
        float *chptrs[128];

        void execute() {
            auto entry = (void (*)())prog->mem;
            asm volatile (
                "call *%0"
                :
                : "" (entry)
                , "c" ((uintptr_t)(void *)prog->consts)
                , "d" ((uintptr_t)(void *)chptrs)
                , "b" ((uintptr_t)(void *)locals)
                : "cc", "memory"
                );
        }

        float *&channel_pointer(int chid) {
            return chptrs[chid];
        }
    };

    inline Context make_context() {
        return {this};
    }

    Program() = default;
    Program(Program const &) = delete;
    ~Program();

    static std::unique_ptr<Program> assemble(std::string const &lines);
};

}
