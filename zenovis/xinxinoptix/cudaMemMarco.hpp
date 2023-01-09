#ifndef __CUDA_MEM_MARCO_HPP__
#define __CUDA_MEM_MARCO_HPP__

#define ENABLE_TRACE_CUDA_MEM 0
#if ENABLE_TRACE_CUDA_MEM

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <link.h>
#include <cudaMemTracer.hpp>
#include <string>

static void systemCall0(const std::string &str)
{
    char command0[256] = {0};
    snprintf(command0, sizeof(command0), "echo \"%s\" >> wjj", str.c_str()); 
    system(command0);
    char command[256] = "nvidia-smi >> wjj";
    system(command);
}

// addr2line -f -e ./build/bin/zenoedit -a <elf>
static void* convToElf(void *addr)
{
    Dl_info info;
    link_map *link;
    dladdr1((void *)addr, &info, (void **)&link, RTLD_DL_LINKMAP);
    return (void *)((std::size_t)addr - link->l_addr);
}

template <class T>
static cudaError_t _cudaMalloc(
    T **devPtr,
    std::size_t size,
    const char devPtrStr[],
    const char sizeStr[],
    const char file[],
    const unsigned int line,
    const char caller[])
{
    auto cudaError = cudaMalloc(devPtr, size);

    auto &tracer = CudaMemTracer::getInstance();
    tracer.pushMemInfo(
        *devPtr,
        size,
        "cudaMalloc( " + std::string(devPtrStr) + ", " + std::string(sizeStr) + " );",
        caller,
        file,
        line,
        convToElf(__builtin_return_address(0)),
        convToElf(__builtin_return_address(1)),
        convToElf(__builtin_return_address(2)),
        convToElf(__builtin_return_address(3))
    );

    return cudaError;
}

static cudaError_t _cudaMallocArray(
    cudaArray_t *array,
    const struct cudaChannelFormatDesc *desc,
    std::size_t width,
    std::size_t height,
    unsigned int flags,
    const char arrayStr[],                                                                 
    const char descStr[],                                                                  
    const char widthStr[],                                                                
    const char heightStr[],                                                                
    const char flagsStr[],                                                                
    const char file[],
    const unsigned int line,
    const char caller[])
{
    auto cudaError = cudaMallocArray(array, desc, width, height, flags);

    auto &tracer = CudaMemTracer::getInstance();
    tracer.pushMemInfo(
        *array,
        (desc->x + desc->y + desc->z + desc->w) * width * height,
        "cudaMallocArray( " + std::string(arrayStr) + ", " + std::string(descStr) + ", " + std::string(widthStr) + ", " + std::string(heightStr) + ", " + std::string(flagsStr) + " );",
        caller,
        file,
        line,
        convToElf(__builtin_return_address(0)),
        convToElf(__builtin_return_address(1)),
        convToElf(__builtin_return_address(2)),
        convToElf(__builtin_return_address(3))
    );

    return cudaError;
}

static cudaError_t _cudaFree(
    void *devPtr,
    const char devPtrStr[],
    const char file[],
    const unsigned int line,
    const char caller[])
{
    auto cudaError = cudaFree(devPtr);

    auto &tracer = CudaMemTracer::getInstance();
    tracer.popMemInfo(devPtr);

    return cudaError;
}

#define cudaMalloc(devPtr, size) _cudaMalloc( \
    devPtr,                                   \
    size,                                     \
    #devPtr,                                  \
    #size,                                    \
    __FILE__,                                 \
    __LINE__,                                 \
    __FUNCTION__)

#define cudaMallocArray(array, desc, width, height, flags) _cudaMallocArray( \
    array,                                                                   \
    desc,                                                                    \
    width,                                                                   \
    height,                                                                  \
    flags,                                                                   \
    #array,                                                                  \
    #desc,                                                                   \
    #width,                                                                  \
    #height,                                                                 \
    #flags,                                                                  \
    __FILE__,                                                                \
    __LINE__,                                                                \
    __FUNCTION__)

#define cudaFree(devPtr) _cudaFree( \
    devPtr,                         \
    #devPtr,                        \
    __FILE__,                       \
    __LINE__,                       \
    __FUNCTION__)

#endif

#endif
