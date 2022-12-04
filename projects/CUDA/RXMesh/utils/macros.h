#pragma once

#include <zeno/utils/log.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>

namespace zeno::rxmesh {

typedef uint8_t flag_t;

// TRANSPOSE_ITEM_PER_THREAD
constexpr uint32_t TRANSPOSE_ITEM_PER_THREAD = 9;

// unsigned 64-bit
#define INVALID64 0xFFFFFFFFFFFFFFFFu

// unsigned 32-bit
#define INVALID32 0xFFFFFFFFu

// unsigned 16-bit
#define INVALID16 0xFFFFu

// http://www.decompile.com/cpp/faq/file_and_line_error_string.htm
#define STRINGIFY(x) TOSTRING(x)
#define TOSTRING(x) #x

// CUDA_ERROR
inline void HandleError(cudaError_t err, const char* file, int line)
{
    // Error handling micro, wrap it around function whenever possible
    if (err != cudaSuccess) {
#ifdef _WIN32
        system("pause");
#else
        exit(EXIT_FAILURE);
#endif
    }
}
#define CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// GPU_FREE
#define GPU_FREE(ptr)              \
    if (ptr != nullptr) {          \
        CUDA_ERROR(cudaFree(ptr)); \
        ptr = nullptr;             \
    }

// Taken from https://stackoverflow.com/a/12779757/1608232
#if defined(__CUDACC__)  // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif

// Taken from
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-traits
#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

}  // namespace zeno