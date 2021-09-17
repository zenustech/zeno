#pragma once

#include <cstdlib>
#include <cstring>
#include "Dim3.h"

namespace ImplHost {

struct Allocator {
    void *allocate(size_t n) {
        return std::malloc(n);
    }

    void *zeroallocate(size_t n) {
        return std::calloc(n, 1);
    }

    void *reallocate(void *old_p, size_t old_n, size_t new_n) {
        return std::realloc(old_p, new_n);
    }

    void deallocate(void *p) {
        std::free(p);
    }
};

using DeviceAllocator = Allocator;

struct Queue {
    template <class Kernel>
    void parallel_for(Dim3 dim, Kernel kernel) {
        for (size_t z = 0; z < dim.z; z++) {
            for (size_t y = 0; y < dim.y; y++) {
                for (size_t x = 0; x < dim.x; x++) {
                    kernel(Dim3(x, y, z));
                }
            }
        }
    }

    void memcpy_dtod(void *d1, void *d2, size_t size) {
        std::memcpy(d1, d2, size);
    }

    void memcpy_dtoh(void *h, void *d, size_t size) {
        std::memcpy(h, d, size);
    }

    void memcpy_htod(void *h, void *d, size_t size) {
        std::memcpy(d, h, size);
    }

    Allocator allocator() {
        return {};
    }

    DeviceAllocator device_allocator() {
        return {};
    }
};

template <class T>
T &make_atomic_ref(T &t) {
    return t;
}

}
