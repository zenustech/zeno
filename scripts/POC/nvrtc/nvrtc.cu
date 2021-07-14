#include <nvrtc_helper.h>
#include <cassert>
#include <cstdio>
#include <cmath>

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
    CUmodule hmod;
    char *data;
    size_t size = 0;

    explicit CudaModule(const char *source) {
        compileSourceToCUBIN("<module>", source, 0, NULL, &data, &size, 0);
        hmod = loadCUBIN(data, 0, NULL);
    }

    CudaFunction getFunction(const char *name) {
        CudaFunction func;
        checkCudaErrors(cuModuleGetFunction(&func.hfunc, hmod, name));
        return func;
    }
};

__global__ void wrangler(void (*func)()) {
    func();
}

int main(int argc, char **argv)
{
    CudaModule mod("extern \"C\" __global__ void callee() {"
                   "printf(\"THIS IS A FUCKING NVIDIA FUNC\\n\"); }");

    auto callee = mod.getFunction("callee");
    //callee.launch();

    checkCudaErrors(cuCtxSynchronize());

    return 0;
}
