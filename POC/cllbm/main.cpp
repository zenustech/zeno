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
kernel void updateGlobal(__read_write image2d_t img, int nx, int ny) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float val = sin(4.f * (float)x / nx);
    write_imagef(img, (int2)(x, y), val);
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

    cl::Image2D img(cl::Context::getDefault(),
            CL_MEM_READ_WRITE, {CL_R, CL_FLOAT}, nx, ny);

    cl::KernelFunctor<
        cl::Image2D const &, int, int
    > kernel(program, "updateGlobal");

    kernel(cl::NDRange(nx, ny), img, nx, ny).wait();

    auto h_img = new float[nx * ny];

    cl::enqueueReadImage(img, true, {0, 0, 0}, {nx, ny, 1}, 0, 0, h_img);
    stbi_write_hdr("/tmp/a.hdr", nx, ny, 1, h_img);
    system("display /tmp/a.hdr");

    return 0;
}
