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
        kernel(threadIdx);
    });
}

int main() {
    parallelFor(dim3(1, 1, 1), dim3(1, 1, 4), [=] __device__ (dim3 threadIdx) {
        printf("hello, world! %d\n", threadIdx.z);
    });
    parallelFor(dim3(1, 1, 1), dim3(2, 1, 1), [=] __device__ (dim3 threadIdx) {
        printf("are you ok? %d\n", threadIdx.x);
    });

    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
