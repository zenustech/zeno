#pragma once // vim: ft=cuda

#define FDB_IMPL_CUDA 1
#define FDB_CONSTEXPR constexpr __host__ __device__
#define FDB_HOST_DEVICE __host__ __device__
#define FDB_DEVICE __device__

#include "helper_cuda.h"
#include <utility>
#include "vec.h"

namespace fdb {

static void synchronize() {
    checkCudaErrors(cudaDeviceSynchronize());
}

template <class Kernel>
static __global__ void __parallel_for(Kernel kernel) {
    kernel();
}

template <class Kernel>
static void parallelGridBlockFor(vec3S grid_dim, vec3S block_dim, Kernel kernel) {
    dim3 gridDim(grid_dim[0], grid_dim[1], grid_dim[2]);
    dim3 blockDim(block_dim[0], block_dim[1], block_dim[2]);
    __parallel_for<<<gridDim, blockDim>>>([=] __device__ () {
        vec3S block_idx(blockIdx.x, blockIdx.y, blockIdx.z);
        vec3S thread_idx(threadIdx.x, threadIdx.y, threadIdx.z);
        kernel(std::as_const(block_idx), std::as_const(thread_idx));
    });
}

struct ParallelConfig {
    size_t block_size;
    size_t saturation;
};

template <class Kernel>
static void parallel_for(vec3S dim, Kernel kernel, ParallelConfig cfg = {8, 1}) {
    vec3S block_dim = clamp(dim, 1, max(1, cfg.block_size));
    vec3S grid_dim = (dim + block_dim - 1) / (block_dim * max(1, cfg.saturation));
    dim3 gridDim(grid_dim[0], grid_dim[1], grid_dim[2]);
    dim3 blockDim(block_dim[0], block_dim[1], block_dim[2]);
    __parallel_for<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t z = blockDim.z * blockIdx.z + threadIdx.z; z < dim[2]; z += gridDim.z * blockDim.z) {
            for (size_t y = blockDim.y * blockIdx.y + threadIdx.y; y < dim[1]; y += gridDim.y * blockDim.y) {
                for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim[0]; x += gridDim.x * blockDim.x) {
                    vec3S idx(x, y, z);
                    kernel(std::as_const(idx));
                }
            }
        }
    });
}

template <class Kernel>
static void parallel_for(vec2S dim, Kernel kernel, ParallelConfig cfg = {32, 1}) {
    vec2S block_dim = clamp(dim, 1, max(1, cfg.block_size));
    vec2S grid_dim = (dim + block_dim - 1) / (block_dim * max(1, cfg.saturation));
    dim3 gridDim(grid_dim[0], grid_dim[1], 1);
    dim3 blockDim(block_dim[0], block_dim[1], 1);
    __parallel_for<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t y = blockDim.y * blockIdx.y + threadIdx.y; y < dim[1]; y += gridDim.y * blockDim.y) {
            for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim[0]; x += gridDim.x * blockDim.x) {
                vec3S idx(x, y);
                kernel(std::as_const(idx));
            }
        }
    });
}

template <class Kernel>
static void parallel_for(size_t dim, Kernel kernel, ParallelConfig cfg = {1024, 1}) {
    size_t block_dim = clamp(dim, 1, max(1, cfg.block_size));
    size_t grid_dim = max(1, (dim + block_dim - 1) / (block_dim * max(1, cfg.saturation)));
    dim3 gridDim(grid_dim, 1, 1);
    dim3 blockDim(block_dim, 1, 1);
    __parallel_for<<<gridDim, blockDim>>>([=] __device__ () {
        for (size_t x = blockDim.x * blockIdx.x + threadIdx.x; x < dim; x += gridDim.x * blockDim.x) {
            kernel(std::as_const(x));
        }
    });
}

static void memcpy_d2d(void *dst, const void *src, size_t n) {
    checkCudaErrors(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice));
}

static void memcpy_d2h(void *dst, const void *src, size_t n) {
    checkCudaErrors(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

static void memcpy_h2d(void *dst, const void *src, size_t n) {
    checkCudaErrors(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

/*static void *allocate_shared(size_t n) {
    void *p = nullptr;
    checkCudaErrors(cudaMallocManaged(&p, n));
    return p;
}*/

static void *allocate(size_t n) {
    void *p = nullptr;
    checkCudaErrors(cudaMalloc(&p, n));
    return p;
}

static void deallocate(void *p) {
    checkCudaErrors(cudaFree(p));
}

static void *reallocate(void *p, size_t old_n, size_t new_n) {
    void *new_p = allocate(new_n);
    memcpy_d2d(new_p, p, std::min(old_n, new_n));
    deallocate(p);
    return new_p;
}

template <class T>
static __device__ T atomic_cas(T *dst, T cmp, T src) {
    return atomicCAS(dst, cmp, src);
}

template <class T>
static __device__ T atomic_add(T *dst, T src) {
    return atomicAdd(dst, src);
}

template <class T>
static __device__ T atomic_sub(T *dst, T src) {
    return atomicSub(dst, src);
}

template <class T>
static __device__ T atomic_max(T *dst, T src) {
    return atomicMax(dst, src);
}

template <class T>
static __device__ T atomic_min(T *dst, T src) {
    return atomicMin(dst, src);
}

template <class T>
static __device__ T atomic_load(T const *src) {
    const volatile T *vaddr = src;
    __threadfence();
    const T value = *vaddr;
    __threadfence();
    return value;
}

template <class T>
static __device__ void atomic_store(T *dst, T src) {
    const volatile T *vaddr = dst;
    __threadfence();
    *vaddr = src;
}

}
