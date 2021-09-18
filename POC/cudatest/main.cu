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
__global__ void __parallelFor(Kernel kernel) {
    kernel();
}

template <class Kernel>
void parallelFor(dim3 gridDim, dim3 blockDim, Kernel kernel) {
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        kernel();
    });
};

int main() {
    parallelFor(dim3(1, 1, 1), dim3(1, 1, 4), [=] __device__ () {
        printf("hello, world!\n");
    });
    parallelFor(dim3(1, 1, 1), dim3(1, 1, 2), [=] __device__ () {
        printf("are you ok?\n");
    });

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
