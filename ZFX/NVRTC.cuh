#include <cuda.h>
#include <nvrtc.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

// PTX codes are generated with the following command:
//   nvcc kernel.cu -ptx -arch=sm_75 -o kernel.ptx
// may also add `--keep` to CMAKE_CUDA_FLAGS to obtain .ptx files in build/
static std::vector<std::string> getAllPTXFilesUnder(std::string const &dirpath) {
    std::vector<std::string> res;

    for (auto const &entry: fs::directory_iterator(dirpath)) {
        auto path = entry.path();
        if (fs::path(path).extension() == ".ptx") {
            printf("reading ptx file: %s\n", path.c_str());
            std::ifstream fin(path,
                std::ios::in | std::ios::binary | std::ios::ate);
            if (!fin.is_open()) {
                std::cerr << "\nerror: unable to open "
                    << path << " for reading!\n";
                abort();
            }

            size_t inputSize = (size_t)fin.tellg();
            char *memBlock = new char[inputSize + 1];

            fin.seekg(0, std::ios::beg);
            fin.read(memBlock, inputSize);
            fin.close();

            memBlock[inputSize] = '\0';
            res.emplace_back(memBlock);
            delete memBlock;
        }
    }
    return res;
}

#define CU(x) do { \
    CUresult __res = (x); \
    if (__res != CUDA_SUCCESS) { \
        const char *__err; \
        cuGetErrorString(__res, &__err); \
        printf("%s:%d: %s: %s (%d)\n", __FILE__, __LINE__, #x, __err, __res); \
        abort(); \
    } \
} while (0)

#define NVRTC(x) do { \
    nvrtcResult __res = (x); \
    if (__res != NVRTC_SUCCESS) { \
        const char *__err = nvrtcGetErrorString(__res); \
        printf("%s:%d: %s: %s (%d)\n", __FILE__, __LINE__, #x, __err, __res); \
        abort(); \
    } \
} while (0)

static CUmodule compileJITModule
    ( CUdevice dev
    , const char *source
    , std::vector<std::string> const &existingPtxs
    ) {
    int major, minor;
    CU(cuDeviceGetAttribute(&major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU(cuDeviceGetAttribute(&minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

    printf("arch=sm_%d%d\n", major, minor);

    char compileArchParam[256];
    sprintf(compileArchParam, "--gpu-architecture=sm_%d%d", major, minor);
    char *compileParams[] = {compileArchParam, (char *)"--device-c"};

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, source, "<nvrtc>", 0, NULL, NULL);
    nvrtcResult res = nvrtcCompileProgram(prog, 2, compileParams);

    size_t logSize;
    NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize + 1];
    NVRTC(nvrtcGetProgramLog(prog, log));
    log[logSize] = '\0';
    if (strlen(log) >= 2) {
        std::cerr << "\n compilation log ---\n";
        std::cerr << log;
        std::cerr << "\n end log ---\n";
    }
    delete log;

    size_t codeSize;
    NVRTC(nvrtcGetPTXSize(prog, &codeSize));
    char *code = new char[codeSize + 1];
    NVRTC(nvrtcGetPTX(prog, code));
    code[codeSize] = '\0';

    printf("\n compilation result ---\n%s\n end result ---\n", code);

    CUlinkState state;
    CU(cuLinkCreate(0, NULL, NULL, &state));

    CU(cuLinkAddData(state, CU_JIT_INPUT_PTX, code,
        codeSize, "nvrtc", 0, NULL, NULL));
    delete code;

    for (auto const &ptx: existingPtxs) {
        CU(cuLinkAddData(state, CU_JIT_INPUT_PTX, (char *)ptx.data(),
            ptx.size(), "kernel", 0, NULL, NULL));
    }

    void *cubin;
    size_t cubinSize;
    CU(cuLinkComplete(state, &cubin, &cubinSize));

    CUmodule module;
    CU(cuModuleLoadData(&module, cubin));

    return module;
}


#if 0
int main() {
    CU(cuInit(0));

    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));

    CUcontext ctx;
    CU(cuCtxCreate(&ctx, 0, dev));

    CUmodule module = compileJITModule(dev, source,
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
#endif
