
#include "utilities.h"

namespace gpu {
//TODO : do not use cuda functions outside this directory.
void gpu_copy(char* dst, char * src, size_t size)
{
    //TODO use a compiling tag to switch between gpu buffer type
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void gpu_to_cpu(char* dst, char * src, size_t size)
{
    //TODO use a compiling tag to switch between gpu buffer type
    checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
}

void cpu_to_gpu(char* dst, char * src, size_t size)
{
    //TODO use a compiling tag to switch between gpu buffer type
    checkCudaErrors(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

void gpu_malloc(void ** buffer, size_t size)
{
    //TODO use a compiling tag to switch between gpu buffer type
    cudaMalloc(buffer, size);
}

void freeGPUBuffer(void* buffer)
{

    cudaFree(buffer);
}


//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}
}