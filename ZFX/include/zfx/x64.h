#pragma once

#include <memory>
#include <cstring>

namespace zfx::x64 {

struct Program {
    float consts[4096];
    uint8_t *mem = nullptr;
    size_t memsize = 0;

    static constexpr size_t SimdWidth = 4;

    struct Context {
        Program *prog;
        float locals[SimdWidth * 128];

        void execute() {
            auto entry = (void (*)())prog->mem;
            asm volatile (
                "call *%0"
                :
                : "" (entry)
                , "c" ((uintptr_t)(void *)prog->consts)
                , "d" ((uintptr_t)(void *)locals)
                : "cc", "memory"
                );
        }

        float *pointer(int chid) {
            return locals + SimdWidth * chid;
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
