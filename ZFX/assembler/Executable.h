#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <vector>
#include <tuple>

class ExecutableArena {
protected:
    uint8_t *mem;
    size_t memsize;

public:
    explicit ExecutableArena(size_t memsize_) : memsize(memsize_) {
        mem = (uint8_t *)mmap(NULL, memsize,
            PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0);
        if (mem == MAP_FAILED) {
            perror("mmap");
            abort();
        }
    }

    ~ExecutableArena() {
        munmap(mem, memsize);
    }
};

class ExecutableInstance : public ExecutableArena {
public:
    explicit ExecutableInstance(std::vector<uint8_t> const &insts)
        : ExecutableArena((insts.size() + 4095) / 4096 * 4096) {
        for (size_t i = 0; i < insts.size(); i++) {
            mem[i] = insts[i];
        }
    }

    ExecutableInstance(ExecutableInstance const &) = delete;

    inline void operator()() {
        auto entry = (void (*)())this->mem;
        entry();
    }

    inline std::tuple
        < uintptr_t
        , uintptr_t
        , uintptr_t
        > operator()
        ( uintptr_t rax
        , uintptr_t rcx
        , uintptr_t rdx
        ) {
        auto entry = (void (*)())this->mem;
        asm volatile (
            "call *%6"
            : "=a" (rax), "=c" (rcx), "=d" (rdx)
            : "a" (rax), "c" (rcx), "d" (rdx)
            , "" (entry)
            : "cc", "memory"
            );
        return {rax, rcx, rdx};
    }
};
