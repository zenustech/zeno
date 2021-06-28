#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>


int main(void)
{
  int nx = 32;
  float *img = nullptr;
  checkCudaErrors(cudaMallocManaged(&img, nx * sizeof(float)));

  for (int i = 0; i < nx; i++) {
    img[i] = drand48();
  }

  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < nx; i++) {
    printf("%f\n", img[i]);
  }

  return 0;
}
#endif
