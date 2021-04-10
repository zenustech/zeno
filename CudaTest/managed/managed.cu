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

const size_t Nx = 128;

template <class T>
__global__ void blur(T arr)
{
  size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < Nx) {
    arr[ix] = ix + 1;
  }
}

int main(void)
{
  Field<Dense<Pointer<Dense<Place<int>, 4, 0>>, Nx / 4, 0>> arr_; auto arr = arr_.copy();

  for (size_t ix = 0; ix < Nx; ix++) {
    arr.activate(ix);
    arr[ix] = 3;
  }
  blur<<<(Nx + 1023) / 1024, (Nx < 1024 ? Nx : 1024)>>>(arr);
  checkCudaErrors(cudaDeviceSynchronize());
  for (size_t ix = 0; ix < Nx; ix++) {
    printf("%d\n", arr[ix]);
  }

  return 0;
}
#endif
