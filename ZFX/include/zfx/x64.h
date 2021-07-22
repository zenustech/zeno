#pragma once

#include <memory>
#include <cstring>
#include <string>
#include <map>

namespace zfx::x64 {

struct Executable {
    uint8_t *mem = nullptr;
    size_t memsize = 0;
    float consts[1024];
    void **functable = nullptr;

    static constexpr size_t SimdWidth = 4;

    struct Context {
        Executable *exec;
        float locals[SimdWidth * 256];

        void execute() {
            // Linux: https://stackoverflow.com/questions/18024672/what-registers-are-preserved-through-a-linux-x86-64-function-call
            // Windows: https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-160#parameter-passing
            // TL;DR: Linux use RDI, RSI, RDX, RCX, R8, R9; Windows use RCX, RDX, R8, R9
#if defined(_WIN32)
            auto entry = (void(*)(void *, void *))exec->mem;
            entry((void*)&locals[0], (void*)&exec->consts[0]);
#else
            auto rax_val = (uintptr_t)(void*)&exec->mem;
            auto rbx_val = (uintptr_t)(void*)exec->functable;
            auto rsi_val = (uintptr_t)(void*)&exec->consts[0];
            auto rdi_val = (uintptr_t)(void*)&locals[0];
            asm volatile (
                "call *(%%rax)"  // why `call *%%rax` gives CE...
                :
                : "a" (rax_val)
                , "b" (rbx_val)
                , "S" (rsi_val)
                , "D" (rdi_val)
                : "cc", "memory"
                );
#endif
        }

        float *channel(int chid) {
            return locals + SimdWidth * chid;
        }
    };

    inline float &parameter(int parid) {
        return consts[parid];
    }

    inline Context make_context() {
        return {this};
    }

    Executable() = default;
    Executable(Executable const &) = delete;
    ~Executable();

    static std::unique_ptr<Executable> assemble
        ( std::string const &lines
        );
};

struct Assembler {
    std::map<std::string, std::unique_ptr<Executable>> cache;

    Executable *assemble(std::string const &lines) {
        if (auto it = cache.find(lines); it != cache.end()) {
            return it->second.get();
        }
        auto prog = Executable::assemble(lines);
        auto raw_ptr = prog.get();
        cache[lines] = std::move(prog);
        return raw_ptr;
    }
};

}
