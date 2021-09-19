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

//static void reallocate(void *p, size_t old_n, size_t new_n) {
//}

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

template <size_t block_width = 8, size_t grid_dim_scale = 1, class Kernel>
static void parallelFor(vec3S dim, Kernel kernel) {
    vec3S block_dim = clamp(dim, 1, max(1, block_width));
    vec3S grid_dim = (dim + block_dim - 1) / (block_dim * max(1, grid_dim_scale));
    dim3 gridDim(grid_dim[0], grid_dim[1], grid_dim[2]);
    dim3 blockDim(block_dim[0], block_dim[1], block_dim[2]);
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t z = blockDim.z * blockIdx.z + threadIdx.z; z < dim[2]; z += gridDim.z * blockDim.z) {
            for (size_t y = blockDim.y * blockIdx.y + threadIdx.y; y < dim[1]; y += gridDim.y * blockDim.y) {
                for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim[0]; x += gridDim.x * blockDim.x) {
                    vec3S idx(x, y, z);
                    kernel(idx);
                }
            }
        }
    });
}

template <size_t block_width = 32, size_t grid_dim_scale = 1, class Kernel>
static void parallelFor(vec2S dim, Kernel kernel) {
    vec2S block_dim = clamp(dim, 1, max(1, block_width));
    vec2S grid_dim = (dim + block_dim - 1) / (block_dim * max(1, grid_dim_scale));
    dim3 gridDim(grid_dim[0], grid_dim[1], 1;
    dim3 blockDim(block_dim[0], block_dim[1], 1);
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t y = blockDim.y * blockIdx.y + threadIdx.y; y < dim[1]; y += gridDim.y * blockDim.y) {
            for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim[0]; x += gridDim.x * blockDim.x) {
                vec3S idx(x, y);
                kernel(idx);
            }
        }
    });
}

template <size_t block_width = 1024, size_t grid_dim_scale = 1, class Kernel>
static void parallelFor(size_t dim, Kernel kernel) {
    size_t block_dim = clamp(dim, 1, max(1, block_width));
    size_t grid_dim = (dim + block_dim - 1) / (block_dim * max(1, grid_dim_scale));
    dim3 gridDim(grid_dim, 1, 1);
    dim3 blockDim(block_dim, 1, 1);
    __parallelFor<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim[0]; x += gridDim.x * blockDim.x) {
            kernel(x);
        }
    });
}

static void synchronize() {
    checkCudaErrors(cudaDeviceSynchronize());
}

#define FDB_DEVICE __device__

}
