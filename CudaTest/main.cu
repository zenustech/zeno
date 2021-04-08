#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include "ndarray.cuh"
#include <cassert>
#include <cstdio>
#include <cmath>

int main(void)
{
  NDArray arr({16}, {sizeof(int)});

  for (ssize_t ix = 0; ix < 16; ix++) {
    *(int *)arr({ix}) = 3;
  }

  blur<<<4, 4>>>(arr);
  checkCudaErrors(cudaDeviceSynchronize());
  for (ssize_t ix = 0; ix < 16; ix++) {
    printf("%d\n", *(int *)arr({ix}));
  }

  return 0;
}
#endif
