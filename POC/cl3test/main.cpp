#if !defined(__OPENCL__)
#include <CL/opencl.hpp>
#include <vector>
#include <memory>
#include <cstdio>
#endif

#if defined(__OPENCL__)
__kernel void test() {
    printf("hello from kernel!\n");
}
#endif

#if !defined(__OPENCL__)
void __CLK_test(NDRrange range) {
    static std::unique_ptr<cl::Kernel> kern;
    if (!kern) kern = std::make_unique<Kernel>(program, "test");
    cl::KernelFunctor func(kern);
    func(range);
}
#endif

#if !defined(__OPENCL__)
std::vector<std::vector<unsigned char>> __CLRT_binaries;
std::unique_ptr<cl::Program> __CLRT_program;

void __CLRT_init() {
    cl::Context context = cl::Context::getDefault();
    std::vector<cl::Device> devices = {cl::Device::getDefault()};
    cl::Program::Binaries binaries;
    for (auto spir: __CLRT_binaries) {
        binaries.push_back(spir);
    }
    __CLRT_program = std::make_unique<cl::Program>(context, devices, binaries);
}
#endif

#if !defined(__OPENCL__)
int main() {
    __CLRT_init();
    return 0;
}
#endif
