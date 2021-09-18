// https://www.slideserve.com/lars/3d-simulation-of-particle-motion-in-lid-driven-cavity-flow-by-mrt-lbm
#include "helper_cuda.h"
#include "helper_math.h"
#include <cstdio>

template <typename T>
struct Allocator {
    T *allocate(size_t n) {
        T *p;
        checkCudaErrors(cudaMallocManaged(&p, n * sizeof(T)));
    }

    void deallocate(T *p) {
        checkCudaErrors(cudaFree(p));
    }
};

template <class Kernel>
__global__ void __launch(Kernel kernel) {
    kernel();
}

template <class Kernel>
void launch(Kernel kernel) {
    __launch<<<1, 1>>>([=] __device__ () {
        kernel();
    });
};

int main() {
    launch([=] __device__ () {
        printf("hello, world!\n");
    });

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
