#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <memory>

int main(void)
{
    std::string kernel1 = R"CLC(
kernel void updateGlobal(global float *img, int nx, int ny) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    printf("before %d, %d: %f\n", x, y, img[x +nx* y]);
    img[x +nx* y] = 3.14f;
    printf("after  %d, %d: %f\n", x, y, img[x +nx* y]);
}
    )CLC";

    auto ctx = cl::Context::getDefault();

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

    const int nx = 4, ny = 4;

    cl::Buffer img(CL_MEM_READ_WRITE, nx * ny * sizeof(float));

    cl::KernelFunctor<
        cl::Buffer const &, int, int
    > kernel(program, "updateGlobal");

    kernel(cl::NDRange(nx, ny), img, nx, ny).wait();

    auto d_img = new float[nx * ny];

    cl::enqueueReadBuffer(img, true, 0, nx * ny * sizeof(float), d_img);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            printf("%f\n", d_img[x +nx* y]);
        }
    }

    return 0;
}
