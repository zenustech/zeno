#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>

template <typename T>
__host__ __device__ void swap(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}

__global__ void blur1(const float *img, float *img_out, int nx)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  img_out[x] = img[x - 1] + img[x] + img[x + 1];
}

__global__ void blur2(const float *img, float *img_out, int nx)
{
  const float fac[] = {1, 8, 36, 112, 266, 504, 784, 1016, 1107, 1016, 784, 504, 266, 112, 36, 8, 1};
  const int Nfac = sizeof(fac) / sizeof(fac[0]);

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  float res = 0.0f;
  for (int i = 0; i < Nfac; i++) {
    res += fac[i] * img[x + i - Nfac / 2];
  }

  img_out[x] = res;
}

__global__ void blur3(const float *img, float *img_out, int nx)
{
  const float fac[] = {1, 8, 36, 112, 266, 504, 784, 1016, 1107, 1016, 784, 504, 266, 112, 36, 8, 1};
  const int Nfac = sizeof(fac) / sizeof(fac[0]);

  int off = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float mem[512 * 3 * 2];

  for (int i = threadIdx.x; i < 1536; i += 512) {
    mem[i] = img[off - 512 + i];
  }

  __syncthreads();

  float *p = mem, *q = mem + 1536;
  for (int t = 8 - 1; t >= 0; t--) {
    for (int i = 512 - t + threadIdx.x; i < 1024 + t; i += 512) {

      float res = 0.0f;
      for (int j = 0; j < Nfac; j++) {
        res += fac[j] * p[i + j - Nfac / 2];
      }
      q[i] = res;

    }
    swap(p, q);
  }

  __syncthreads();

  img_out[off] = p[512 + threadIdx.x];
}

int main(void)
{
  int nx = 4096 * 4096;
  float *img = nullptr, *img_out = nullptr;
  checkCudaErrors(cudaMallocManaged(&img, nx * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&img_out, nx * sizeof(float)));
  assert(img && img_out);

  for (int i = 0; i < nx; i++) {
    img[i] = drand48();
  }

  for (int i = 0; i < 1024; i++) {
    blur3<<<8192, 512>>>(img, img_out, nx);
    swap(img, img_out);
  }
  checkCudaErrors(cudaDeviceSynchronize());

#if 0
  for (int i = 0; i < 64; i++) {
    printf("%f\n", img[i]);
  }
#endif

  return 0;
}
#endif
