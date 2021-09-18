#pragma once // vim: ft=cuda

#include "helper_cuda.h"
#include "vec.h"

namespace fdb {

void *allocate(size_t n) {
    void *p = nullptr;
    checkCudaErrors(cudaMallocManaged(&p, n));
    return p;
}

void deallocate(void *p) {
    checkCudaErrors(cudaFree(p));
}

template <class Kernel>
__global__ void __parallelFor(Kernel kernel) {
    kernel();
}

template <class Kernel>
void parallelFor(vec3S grid_dim, vec3S block_dim, Kernel kernel) {
    dim3 gridDim(vec_to_other<dim3>(grid_dim));
    dim3 blockDim(vec_to_other<dim3>(block_dim));
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        vec3S block_idx = other_to_vec<3>(blockIdx);
        vec3S thread_idx = other_to_vec<3>(threadIdx);
        kernel(block_idx, thread_idx);
    });
}

#define FDB_DEVICE_FUNCTOR __device__

}
