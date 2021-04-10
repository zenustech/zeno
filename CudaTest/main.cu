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

__global__ void CUDA_blur(NDTypedView<int> arr)
{
  NDDim idx = getIdx();
  NDDim thrIdx = getThreadIdx();
  NDDim blkDim = getBlockDim();

  extern __shared__ char tmpData[];
  NDTypedView<int> tmp({blkDim[0], blkDim[1]}, tmpData);

  tmp({thrIdx[0], thrIdx[1]}) = arr({idx[0], idx[1]});

  //...

  arr({idx[0], idx[1]}) = tmp({thrIdx[0], thrIdx[1]});
}

void blur(NDTypedView<int> arr)
{
  size_t bnx = 8, bny = 8;
  size_t nx = arr.shape()[0], ny = arr.shape()[1];
  Launch launch(CUDA_blur, {nx, ny}, {bnx, bny}, bnx * bny * sizeof(int));
  launch(arr);
}

int main(void)
{
  size_t nx = 32, ny = 32;
  NDTypedArray<int> arr({nx, ny});

  for (size_t iy = 0; iy < ny; iy++) {
    for (size_t ix = 0; ix < nx; ix++) {
      arr({ix, iy}) = 233;
    }
  }

  blur(arr);
  checkCudaErrors(cudaDeviceSynchronize());

  for (size_t iy = 0; iy < ny; iy++) {
    for (size_t ix = 0; ix < nx; ix++) {
      printf("%d\n", arr({ix, iy}));
    }
  }

  return 0;
}
#endif
