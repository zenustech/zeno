#include <cstdio>
#include "impl_cuda.h"

using namespace fdb;

__global__ void a() { printf("a\n"); }

int main() {
    parallelFor(vec3S(1, 1, 1), vec3S(1, 1, 4), [=] FDB_DEVICE (vec3S block_idx, vec3S thread_idx) {
        printf("hello, world! %d\n", thread_idx[2]);
    });

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
