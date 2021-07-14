#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "nvrtc_helper.h"
#include <cassert>
#include <cstdio>
#include <cmath>


int main(int argc, char **argv)
{
    char *cubin;
    size_t cubinSize;
    char *kernel_file = sdkFindFilePath("kernel.cu", argv[0]);
    compileFileToCUBIN(kernel_file, argc, argv, &cubin, &cubinSize, 0);

    CUmodule module = loadCUBIN(cubin, argc, argv);
    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "kernel_func"));

    void *args[] = {};
    checkCudaErrors(cuLaunchKernel(kernel_addr, 1, 1, 1, 1, 1, 1,
        0, 0, args, 0));

    checkCudaErrors(cuCtxSynchronize());

    return 0;
}
#endif
