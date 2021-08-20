#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

int main(void)
{
    std::string kernel1 = R"CLC(
        kernel void updateGlobal(int x) {
          printf("Hello, the value is %d\n", x);
        }
    )CLC";

    cl::Program program({kernel1});
    try {
        program.build("-cl-std=CL2.0");
    } catch (...) {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return 1;
    }

    cl::Image2D img;

    cl::KernelFunctor<> kernel(program, "updateGlobal");
    kernel.getKernel().setArg(0, 42);
    kernel(cl::EnqueueArgs(cl::NDRange(1)));

    return 0;
}
