#pragma once

#include <cstdlib>

namespace ImplHost {

struct HostAllocator {
    void *allocate(size_t n) {
        return malloc(n);
    }

    void *zeroallocate(size_t n) {
        return calloc(n, 1);
    }

    void *reallocate(void *old_p, size_t old_n, size_t new_n) {
        return realloc(old_p, new_n);
    }

    void deallocate(void *p) {
        free(p);
    }
};

}
