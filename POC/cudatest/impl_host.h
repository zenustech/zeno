#pragma once

#define FDB_CONSTEXPR constexpr
#define FDB_HOST_DEVICE
#define FDB_DEVICE

#include <cstdlib>
#include "vec.h"

namespace fdb {

void *allocate(size_t n) {
    return malloc(n);
}

void deallocate(void *p) {
    free(p);
}

template <class Kernel>
__global__ void __parallelFor(Kernel kernel) {
    kernel();
}

template <class Kernel>
void parallelFor(vec3S grid_dim, vec3S block_dim, Kernel kernel) {
    for (size_t i = 0; i < grid_dim[0] * grid_dim[1] * grid_dim[2]; i++) {
        size_t x = i % grid_dim[0];
        size_t y = i % grid_dim[1];
    }
}

#define FDB_DEVICE

}
