#pragma once

#include <memory>
#include <cstring>
#include <sstream>

namespace zfx::x64 {

struct Program {
    uint8_t *mem = nullptr;
    size_t memsize = 0;
    float consts[1024];

    static constexpr size_t SimdWidth = 4;

    struct Context {
        Program *prog;
        float locals[SimdWidth * 256];

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

        float &channel(int chid) {
            return locals[SimdWidth * chid];
        }
    };

    void set_constants(std::vector<std::string> const &constants) {
        for (int i = 0; i < constants.size(); i++) {
            std::istringstream(constants[i]) >> consts[i];
        }
    }

    float &parameter(int parid) {
        return consts[parid];
    }

    inline constexpr Context make_context() {
        return {this};
    }

    Program() = default;
    Program(Program const &) = delete;
    ~Program();

    static std::unique_ptr<Program> assemble(std::string const &lines);
};

}
