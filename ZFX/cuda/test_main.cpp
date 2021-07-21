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

    zfx::Compiler compiler;
    zfx::cuda::Assembler assembler;

    zfx::Options opts(zfx::Options::for_cuda);
    // wxl: reassign (making var refs compact) gives better performance,
    // but make specifying harder, may disable them if necessary:
    opts.reassign_channels = false;
    opts.reassign_parameters = false;
    opts.define_symbol("@pos", 1);
    opts.define_param("$dt", 1);
    std::string zfxcode(R"(
if (1)
@pos = sin(@pos + $dt + 2.718)
endif
)");
    auto prog = compiler.compile(zfxcode, opts);
    auto jitcode = assembler.assemble(prog->assembly);

    CUmodule module = compileJITModule(dev, jitcode.c_str(),
        getAllPTXFilesUnder("cuda/"));

    CUfunction function;
    CU(cuModuleGetFunction(&function, module, "zfx_array_wrangle"));

    float h_array[1] = {3.14f};
    CUdeviceptr d_array;
    size_t nsize = 1;
    float h_params[1] = {0.618f};
    CUdeviceptr d_params;
    CU(cuMemAlloc(&d_array, sizeof(h_array)));
    CU(cuMemcpyHtoD(d_array, h_array, sizeof(h_array)));
    CU(cuMemAlloc(&d_params, sizeof(h_params)));
    CU(cuMemcpyHtoD(d_params, h_params, sizeof(h_params)));
    void *arg1 = (void *)d_array;
    void *arg2 = (void *)nsize;
    void *arg3 = (void *)d_params;
    void *args[] = {&arg1, &arg2, &arg3};
    CU(cuLaunchKernel(function,
            1, 1, 1, 1, 1, 1,
            0, 0, args, 0));

    CU(cuCtxSynchronize());
    CU(cuMemcpyDtoH(h_array, d_array, sizeof(h_array)));
    for (int i = 0; i < nsize; i++) {
        printf("%f\n", h_array[i]);
    }

    printf("done\n");
    return 0;
}
