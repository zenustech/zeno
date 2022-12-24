#ifndef __CUDA_MEM_TRACER_HPP__
#define __CUDA_MEM_TRACER_HPP__

#define ENABLE_CUDA_MEM_TRACER 0
#if ENABLE_CUDA_MEM_TRACER

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <link.h>

constexpr char recordDir[] = "./cudaMemRecord";

static void convToCallerInfoAndPush(void *addr, const char recordFile[])
{
    Dl_info info;
    link_map *link;
    dladdr1((void *)addr, &info, (void **)&link, RTLD_DL_LINKMAP);
    void *caller = (void *)((std::size_t)addr - link->l_addr);

    char command[256];
    snprintf(command, sizeof(command), "addr2line -f -e %s -a %p >> %s", info.dli_fname, caller, recordFile);
    system(command);
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

    char recordFile[128] = {0};
    sprintf(recordFile, "%s/%p.mem", recordDir, *devPtr);
    FILE *fp = fopen(recordFile, "w");
    fprintf(fp,
            "devPtr=%p,\nsize=%ld,\ndevPtrStr=\"%s\",\nsizeStr=\"%s\",\nfile=\"%s\",\nline=%d,\ncaller=\"%s\",\n",
            *devPtr, size, devPtrStr, sizeStr, file, line, caller);
    fprintf(fp, "callStackAddr=\n"),
    fflush(fp);
    fclose(fp);
    convToCallerInfoAndPush(__builtin_return_address(0), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(1), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(2), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(3), recordFile);

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

    char recordFile[128] = {0};
    sprintf(recordFile, "%s/%p.mem", recordDir, *array);
    FILE *fp = fopen(recordFile, "w");
    fprintf(fp,
            "array=%p,\nwidth=%ld,\nheight=%ld,\nflags=%d,\narrayStr=\"%s\",\ndescStr=\"%s\",\nwidthStr=\"%s\",\nheightStr=\"%s\",\nflagsStr=\"%s\",\nfile=\"%s\",\nline=%d,\ncaller=\"%s\",\n",
            *array, width, height, flags, arrayStr, descStr, widthStr, heightStr, flagsStr, file, line, caller);
    fprintf(fp, "callStackAddr=\n"),
    fflush(fp);
    fclose(fp);
    convToCallerInfoAndPush(__builtin_return_address(0), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(1), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(2), recordFile);
    convToCallerInfoAndPush(__builtin_return_address(3), recordFile);

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

    char recordFile[128] = {0};
    sprintf(recordFile, "%s/%p.mem", recordDir, devPtr);
    if (unlink(recordFile) < 0)
    {
        printf("double free: %p\n", devPtr);
    }

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