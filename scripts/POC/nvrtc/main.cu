#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <nvrtc_helper.h>
#include <cassert>
#include <cstdio>
#include <cmath>

struct NvrtcCubin {
    char *data;
    size_t size;

    explicit NvrtcCubin(const char *source) {
        compileSourceToCUBIN("<module>", source, 0, NULL, &data, &size, 0);
    }

    ~NvrtcCubin() {
        delete data;
    }
};

struct CudaFunction {
    CUfunction hfunc;

    void launch
        ( void **args = NULL
        , int gridDimX = 1
        , int gridDimY = 1
        , int gridDimZ = 1
        , int blockDimX = 1
        , int blockDimY = 1
        , int blockDimZ = 1
        ) {
        checkCudaErrors(cuLaunchKernel(hfunc,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            0, 0, args, 0));
    }
};

struct CudaModule {
    NvrtcCubin cubin;
    CUmodule hmod;

    explicit CudaModule(const char *source) : cubin(source) {
        hmod = loadCUBIN(cubin.data, 0, NULL);
    }

    CudaFunction getFunction(const char *name) {
        CudaFunction func;
        checkCudaErrors(cuModuleGetFunction(&func.hfunc, hmod, name));
        return func;
    }
};

int main(int argc, char **argv)
{
    CudaModule mod("extern \"C\" __global__ void kernel_func() { printf(\"FUCKING NVIDIA FUNC\\n\"); }");

    auto func = mod.getFunction("kernel_func");
    func.launch();

    checkCudaErrors(cuCtxSynchronize());

    return 0;
}
#endif
