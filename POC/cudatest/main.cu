#include <cstdio>
#include "vec.h"
#include "impl_cuda.h"

using namespace fdb;

int main() {
    parallelFor(vec3S(1, 1, 1), vec3S(1, 1, 4), [=] FDB_DEVICE_FUNCTOR (vec3S block_idx, vec3S thread_idx) {
        printf("hello, world! %d\n", thread_idx[2]);
    });

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
