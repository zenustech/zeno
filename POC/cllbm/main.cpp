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
constant const int directions[][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,-1},{-1,1,1},{1,-1,-1}};
constant const float weights[] = {2.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f, 1.f/72.f};
constant float inv_tau = 1.f / (3.f * 0.005f + 0.5f);

inline float f_eq(int q, float4 v) {
    float eu = v.x * directions[q][0]
             + v.y * directions[q][1]
             + v.z * directions[q][2];
    float uv = v.x * v.x + v.y * v.y + v.z * v.z;
    float term = 1.f + 3.f * eu + 4.5f * eu * eu - 1.5f * uv;
    float feq = weights[q] * v.w * term;
    return feq;
}

kernel void collide
    ( read_only global float4 *vel
    , read_write global float *fie
    , int nx
    , int ny
    , int nz
    ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int xyz = x +nx* y +nx*ny* z;
    float4 v = vel[xyz];
    float uv = v.x * v.x + v.y * v.y + v.z * v.z;
    uv = 1.f - 1.5f * uv;
    for (int q = 0; q < 15; q++) {
        int xyzq = xyz +nx*ny*nz* q;
        float eu = v.x * directions[q][0]
                 + v.y * directions[q][1]
                 + v.z * directions[q][2];
        float term = uv + 3.f * eu + 4.5f * eu * eu;
        float feq = weights[q] * v.w * term;
        fie[xyzq] = fie[xyzq] * (1.f - inv_tau) + feq * inv_tau;
    }
}

kernel void stream
    ( read_write global float *fie
    , int nx
    , int ny
    , int nz
    ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int xyz = x +nx* y +nx*ny* z;
    for (int q = 0; q < 15; q++) {
        int mdx = (x - directions[q][0] + nx)%nx;
        int mdy = (y - directions[q][1] + ny)%ny;
        int mdz = (z - directions[q][2] + nz)%nz;
        int mdxyzq = mdx +nx* mdy +nx*ny* mdz +nx*ny*nz* q;
        int xyzq = xyz +nx*ny*nz* q;
        fie[xyzq] = fie[mdxyzq];
    }
}

kernel void update_macro
    ( write_only global float4 *vel
    , read_only global float *fie
    , int nx
    , int ny
    , int nz
    ) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    float4 v;
    int xyz = x +nx* y +nx*ny* z;
    for (int q = 0; q < 15; q++) {
        int xyzq = xyz +nx*ny*nz* q;
        float f = fie[xyzq];
        v.x += f * directions[q][0];
        v.y += f * directions[q][1];
        v.z += f * directions[q][2];
        v.w += f;
    }
    float mscale = 1.f / max(v.w, 1e-6f);
    v.x *= mscale;
    v.y *= mscale;
    v.z *= mscale;
    vel[xyz] = v;
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

    const int nx = 64, ny = 64, nz = 64;

    cl::Buffer img(CL_MEM_READ_WRITE, nx * ny * sizeof(float));
    cl::Buffer vel(CL_MEM_READ_WRITE, nx * ny * nz * sizeof(float));
    cl::Buffer fie(CL_MEM_READ_WRITE, nx * ny * nz * sizeof(float));

    cl::KernelFunctor
        < cl::Buffer const &
        , cl::Buffer const &
        , int
        , int
        , int
    > collide(program, "collide");

    cl::KernelFunctor
        < cl::Buffer const &
        , int
        , int
        , int
    > stream(program, "stream");

    cl::KernelFunctor
        < cl::Buffer const &
        , cl::Buffer const &
        , int
        , int
        , int
    > update_macro(program, "update_macro");

    collide(cl::NDRange(nx, ny), vel, fie, nx, ny, nz).wait();
    stream(cl::NDRange(nx, ny), fie, nx, ny, nz).wait();
    update_macro(cl::NDRange(nx, ny), vel, fie, nx, ny, nz).wait();

    auto h_img = new float[nx * ny];
    cl::enqueueReadBuffer(img, true, 0, nx * ny * sizeof(float), h_img);
    stbi_write_hdr("/tmp/a.hdr", nx, ny, 1, h_img);
    system("display /tmp/a.hdr");

    return 0;
}
