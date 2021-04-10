#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include "NDArray.cuh"
#include "Launch.cuh"
#include <cassert>
#include <cstdio>
#include <cmath>

__global__ void blur(NDTypedView<int> arr)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  size_t iy = blockIdx.y * blockDim.y + threadIdx.y;

  arr({ix, iy}) = ix + 1;

  //__shared__ char tmpData[16 * sizeof(int)];
  //NDTypedView<int> tmp({16}, {sizeof(int)}, tmpData);
}

int main(void)
{
  size_t nx = 32, ny = 32;
  NDTypedArray<int> arr({nx, ny});

  for (size_t iy = 0; iy < ny; iy++) {
    for (size_t ix = 0; ix < nx; ix++) {
      *(int *)arr({ix, iy}) = 233;
    }
  }

  Launch(blur, {nx, ny}, {8, 8})(arr);
  checkCudaErrors(cudaDeviceSynchronize());

  for (size_t iy = 0; iy < ny; iy++) {
    for (size_t ix = 0; ix < nx; ix++) {
      printf("%d\n", *(int *)arr({ix, iy}));
    }
  }

  return 0;
}
#endif
