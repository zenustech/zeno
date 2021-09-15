#if !defined(__OPENCL__)
#include <cstdio>
#include <CL/opencl.hpp>
#endif

#if defined(__OPENCL__)
__kernel void test() {
    printf("hello from kernel!\n");
}
#endif

#if !defined(__OPENCL__)
int main() {
    cl::Context context = cl::Context::getDefault();
    std::vector<cl::Device> devices = {cl::Device::getDefault()};
    cl::Program::Binaries binaries;
    std::vector<unsigned char> spir;
    binaries.push_back(spir);
    cl::Program program(context, devices, binaries);
    return 0;
}
#endif
