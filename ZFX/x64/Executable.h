#pragma once

#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>

namespace zfx::x64 {

uint8_t *exec_page_alloc(size_t memsize) {
    uint8_t *mem = (uint8_t *)mmap(NULL, memsize,
        PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
        -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap");
        abort();
    }
    return mem;
}

void exec_page_free(uint8_t *mem, size_t memsize) {
    munmap(mem, memsize);
}

}
