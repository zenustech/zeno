#pragma once

#include "helper_cuda.h"
#include "Array.cuh"

using NDDim = Array<size_t, 3>;

template <class FuncT>
class Launch
{
  FuncT const &mFunc;
  dim3 mGridDim;
  dim3 mBlockDim;
  size_t mSharedSize;

  template <class T, class V>
  static __host__ void iDivMin(T x, T y, V &retDiv, V &retMin) {
    if (x <= 0)
      x = 1;
    if (y <= 0)
      y = 1;
    retDiv = (x + y - 1) / y;
    //retMin = x < y ? x : y;
    retMin = y;
  }

public:
  __host__ Launch(FuncT const &func,
      NDDim const &dim, NDDim const &blkDim, size_t sharedSize = 0)
    : mFunc(func), mSharedSize(sharedSize)
  {
    iDivMin(dim[0], blkDim[0], mGridDim.x, mBlockDim.x);
    iDivMin(dim[1], blkDim[1], mGridDim.y, mBlockDim.y);
    iDivMin(dim[2], blkDim[2], mGridDim.z, mBlockDim.z);
  }

  template <class... Args>
  __host__ void operator()(Args &&... args)
  {
    mFunc<<<mGridDim, mBlockDim, mSharedSize>>>(std::forward<Args>(args)...);
  }
};

inline __device__ NDDim getBlockIdx() {
  return {blockIdx.x, blockIdx.y, blockIdx.z};
}

inline __device__ NDDim getThreadIdx() {
  return {threadIdx.x, threadIdx.y, threadIdx.z};
}

inline __device__ NDDim getGridDim() {
  return {gridDim.x, gridDim.y, gridDim.z};
}

inline __device__ NDDim getBlockDim() {
  return {blockDim.x, blockDim.y, blockDim.z};
}

inline __device__ NDDim getDim() {
  return {
    blockDim.x * gridDim.x,
    blockDim.y * gridDim.y,
    blockDim.z * gridDim.z,
  };
}

inline __device__ NDDim getIdx() {
  return {
    threadIdx.x + blockDim.x * blockIdx.x,
    threadIdx.y + blockDim.y * blockIdx.y,
    threadIdx.z + blockDim.z * blockIdx.z,
  };
}
