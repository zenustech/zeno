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
    CU(cuModuleGetFunction(&function, module, "caller"));

    CU(cuLaunchKernel(function,
            1, 1, 1, 1, 1, 1,
            0, 0, NULL, 0));

    CU(cuCtxSynchronize());

    printf("done\n");
    return 0;
}
