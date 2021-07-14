#include <cuda.h>
#include "NVRTC.cuh"
#include <zfx/zfx.h>
#include <zfx/cuda.h>

#define CU(x) do { \
    CUresult __res = (x); \
    if (__res != CUDA_SUCCESS) { \
        const char *__err; \
        cuGetErrorString(__res, &__err); \
        printf("%s:%d: %s: %s (%d)\n", __FILE__, __LINE__, #x, __err, __res); \
        abort(); \
    } \
} while (0)

int main() {
    CU(cuInit(0));

    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CU(cuCtxCreate(&ctx, 0, dev));

    zfx::Compiler<zfx::cuda::Program> compiler;

    zfx::Options opts;
    opts.define_symbol("@pos", 1);
    std::string zfxcode("@pos = 2.718");
    auto prog = compiler.compile(zfxcode, opts);
    auto source = prog->get_codegen_result();

    CUmodule module = compileJITModule(dev, source.c_str(),
        getAllPTXFilesUnder("."));

    CUfunction function;
    CU(cuModuleGetFunction(&function, module, "zfx_array_wrangle"));

    float array[1] = {3.14f};
    size_t size = 1;
    float params[1];
    void *arg1 = (void *)array;
    void *arg2 = (void *)size;
    void *arg3 = (void *)params;
    void *args[] = {&arg1, &arg2, &arg3};
    CU(cuLaunchKernel(function,
            1, 1, 1, 1, 1, 1,
            0, 0, args, 0));

    CU(cuCtxSynchronize());

    printf("done\n");
    return 0;
}
