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
            uintptr_t rax_val((void *)&exec->mem);
            uintptr_t rdx_val((void *)exec->functable);
            uintptr_t rsi_val((void *)&exec->consts[0]);
            uintptr_t rdi_val((void *)&locals[0]);
#if defined(_MSC_VER)
            __asm {
                mov rax, rax_val
                mov rdx, rdx_val
                mov rsi, rsi_val
                mov rdi, rdi_val
                call [rax]
            }
#else
            asm volatile (
                "call *(%%rax)"  // why `call *%%rax` gives CE...
                :
                : "a" (rax_val)
                , "d" (rdx_val)
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
