#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <vector>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include "stb_image_write.h"

int main(void)
{
    std::string kernel1 = R"CLC(
kernel void updateGlobal(global float *img, int nx, int ny) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float val = sin(4.f * (float)x / nx);
    img[x +nx* y] = val;
}
    )CLC";

    cl::Program program({kernel1});
    try {
        program.build("-cl-std=CL2.0");
    } catch (...) {
        cl_int buildErr = CL_SUCCESS;
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            printf("%s\n\n", pair.second.c_str());
        }
        return 1;
    }

    const int nx = 128, ny = 128;

    cl::Buffer img(CL_MEM_READ_WRITE, nx * ny * sizeof(float));

    cl::KernelFunctor<
        cl::Buffer const &, int, int
    > kernel(program, "updateGlobal");

    kernel(cl::NDRange(nx, ny), img, nx, ny).wait();

    auto h_img = new float[nx * ny];

    cl::enqueueReadBuffer(img, true, 0, nx * ny * sizeof(float), h_img);
    stbi_write_hdr("/tmp/a.hdr", nx, ny, 1, h_img);
    system("display /tmp/a.hdr");

    return 0;
}
