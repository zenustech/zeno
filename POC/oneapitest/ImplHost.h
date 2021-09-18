#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Dim3.h"

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

    template <class T>
    struct __AtomicRef {
        T &t;

        __AtomicRef(T &t) : t(t) {}

        inline T load() {
            return t;
        }

        inline void store(T value) {
            t = value;
        }

        bool store_if_equal(T if_equal, T then_set) {
            if (t == if_equal) {
                t = then_set;
                return true;
            }
            return false;
        }

        inline T fetch_inc() {
            return t++;
        }
    };

    template <class T>
    static auto make_atomic_ref(T &t) {
        return __AtomicRef<T>(t);
    }

    template <class ...Args>
    static void printf(Args &&...args) {
        std::printf(std::forward<Args>(args)...);
    }
};
