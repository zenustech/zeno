#if !defined(__OPENCL__)
#include <CL/opencl.hpp>
#include <vector>
#include <memory>
#include <cstdio>
#endif

#if defined(__OPENCL__)
extern "C" __kernel void test() {
    printf("hello from kernel!\n");
}
#endif

#if !defined(__OPENCL__)
std::vector<std::vector<unsigned char>> __CLRT_binaries = {
#include "main.bc.inl"
};
std::unique_ptr<cl::Program> __CLRT_program;

cl::Program *__CLRT_init() {
    if (!__CLRT_program) {
        cl::Context context = cl::Context::getDefault();
        std::vector<cl::Device> devices = {cl::Device::getDefault()};
        cl::Program::Binaries binaries;
        for (auto spir: __CLRT_binaries) {
            binaries.push_back(spir);
        }
        __CLRT_program = std::make_unique<cl::Program>(
                context, devices, binaries);
    }
    return __CLRT_program.get();
}

void __CLK_test(cl::NDRange range) {
    static std::unique_ptr<cl::KernelFunctor<>> kernel;
    if (!kernel)
        kernel = std::make_unique<cl::KernelFunctor<>>(*__CLRT_init(), "test");
    printf("!!!\n");
    (*kernel)(range);
    printf("???\n");
}
#endif

#if !defined(__OPENCL__)
int main() {
    __CLRT_init();
    __CLK_test({1});
    return 0;
}
#endif
