#if 0
#include <stdio.h>
__global__ void test() { printf("FuCK U NVIDIA!\n"); } int main(void) { test<<<1, 1>>>(); cudaDeviceSynchronize(); }
#else

#include "helper_cuda.h"
#include "helper_math.h"
#include <cassert>
#include <cstdio>
#include <cmath>

template <class T, size_t N>
class Array {
  T m_data[N];

public:
  Array() = default;
  Array(Array const &) = default;
  Array(Array &&) = default;

  __host__ __device__ Array(std::initializer_list<T> const &args) {
    int i = 0;
    for (auto const &value: args) {
      m_data[i++] = value;
    }
    for (; i < N; i++) {
      m_data[i] = T(0);
    }
  }

  __host__ __device__ T &operator[](long i) {
    return m_data[i];
  }

  __host__ __device__ T const &operator[](long i) const {
    return m_data[i];
  }
};

struct NDView {
  static constexpr long Dims = 1;
  Array<long, Dims> stride;
  Array<long, Dims> shape;
  void *base{nullptr};

  NDView() = default;
  NDView(NDView const &) = default;
  NDView(NDView &&) = default;

  __host__ __device__ void *operator()(Array<long, Dims> const &indices) {
    long offset = 0;
    for (int i = 0; i < Dims; i++) {
      offset += stride[i] * indices[i];
    }
    return (char *)base + offset;
  }
};

__global__ void blur(NDView arr)
{
  long ix = blockIdx.x * blockDim.x + threadIdx.x;
  *(int *)arr({ix}) = ix + 1;
}

int main(void)
{
  NDView arr;
  arr.shape[0] = 16;
  arr.stride[0] = sizeof(int);
  cudaMallocManaged(&arr.base, arr.shape[0] * arr.stride[0]);
  for (long ix = 0; ix < 16; ix++) {
    *(int *)arr({ix}) = 3;
  }

  blur<<<4, 4>>>(arr);
  checkCudaErrors(cudaDeviceSynchronize());
  for (long ix = 0; ix < 16; ix++) {
    printf("%d\n", *(int *)arr({ix}));
  }

  cudaFree(arr.base);
  return 0;
}
#endif
