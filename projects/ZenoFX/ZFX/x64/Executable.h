#pragma once

#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace zfx::x64 {

void *exec_page_allocate(size_t size) {
#if defined(_WIN32)
    void *ptr = VirtualAlloc(nullptr, size, MEM_COMMIT,
            PAGE_READWRITE);
    if (!ptr) {
        printf("VirtualAlloc failed!\n");
        abort();
    }
#else
    void *ptr = mmap(nullptr, size,
            PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        abort();
    }
#endif
    return ptr;
}

void exec_page_mark_executable(void *ptr, size_t size) {
#if defined(_WIN32)
    DWORD dummy;
    VirtualProtect(ptr, size, PAGE_EXECUTE_READ, &dummy);
#else
// https://stackoverflow.com/questions/40936534/how-to-alloc-a-executable-memory-buffer
#endif
}

void exec_page_free(void *ptr, size_t size) {
#if defined(_WIN32)
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);
#endif
}

}
