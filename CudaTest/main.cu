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

__global__ void blur(MemoryArray<float> *parr)
{
  auto &arr = *parr;
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  printf("gpu:%d\n", (int)arr.size());
  if (ix < arr.size()) {
    arr(ix) = 2 * arr(ix);
  }
}

int main(void)
{
  const size_t nx = 1 * 1;
  auto parr = new MemoryArray<float>(nx);
  auto &arr = *parr;

  for (size_t ix = 0; ix < nx; ix++) {
    arr(ix) = drand48();
  }
  printf("cpu:%d\n", (int)arr.size());

  for (size_t i = 0; i < 256; i++) {
    blur<<<(nx + 1023) / 1024, (nx < 1024 ? nx : 1024)>>>(parr);
  }
  checkCudaErrors(cudaDeviceSynchronize());

  delete parr;
  return 0;
}
#endif
