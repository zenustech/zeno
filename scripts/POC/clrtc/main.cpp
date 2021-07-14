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
        kernel void updateGlobal()
        {
          printf("Hello, World!\n");
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
    cl::KernelFunctor<> kernel(program, "updateGlobal");
    kernel(cl::EnqueueArgs(cl::NDRange(1)));
    return 0;
}
