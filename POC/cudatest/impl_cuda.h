#pragma once // vim: ft=cuda

#include "helper_cuda.h"
#define FDB_VEC_CONSTEXPR constexpr __host__ __device__
#include "vec.h"

namespace fdb {

static void *allocate(size_t n) {
    void *p = nullptr;
    checkCudaErrors(cudaMallocManaged(&p, n));
    return p;
}

static void deallocate(void *p) {
    checkCudaErrors(cudaFree(p));
}

template <class Kernel>
static __global__ void __parallelFor(Kernel kernel) {
    kernel();
}

template <class Kernel>
static void parallelFor(vec3S grid_dim, vec3S block_dim, Kernel kernel) {
    dim3 gridDim(grid_dim[0], grid_dim[1], grid_dim[2]);
    dim3 blockDim(block_dim[0], block_dim[1], block_dim[2]);
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        vec3S block_idx(blockIdx.x, blockIdx.y, blockIdx.z);
        vec3S thread_idx(threadIdx.x, threadIdx.y, threadIdx.z);
        kernel(block_idx, thread_idx);
    });
}

static void synchronize() {
    checkCudaErrors(cudaDeviceSynchronize());
}

#define FDB_DEVICE __device__

}
