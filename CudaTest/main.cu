#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include "managed.cuh"
#include <cassert>
#include <cstdio>
#include <cmath>

template <size_t Nx>
__global__ void blur(Array<float, Nx> *arr)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < Nx) {
    (*arr)(ix) = 2 * (*arr)(ix);
  }
}

int main(void)
{
  const size_t Nx = 16384 * 1024;
  auto arr = new Array<float, Nx>();

  for (size_t x = 0; x < Nx; x++) {
    (*arr)(x) = drand48();
  }

  for (size_t i = 0; i < 256; i++) {
    blur<Nx><<<16384, 1024>>>(arr);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  return 0;
}
#endif
