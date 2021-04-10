#pragma once

#include "helper_cuda.h"
#include "Array.cuh"

template <class FuncT>
class Launch
{
  FuncT const &mFunc;
  dim3 mGridDim;
  dim3 mBlockDim;

  using NDRange = Array<size_t, 3>;

  template <class T>
  static T iDivUp(T x, T y) {
    return (x + y - 1) / y;
  }

  template <class T>
  static T iMin(T x, T y) {
    return x < y ? x : y;
  }

public:
  Launch(FuncT const &func, NDRange dim, NDRange blkDim)
    : mFunc(func)
  {
    if (dim[0] <= 0) dim[0] = 1;
    if (dim[1] <= 0) dim[1] = 1;
    if (dim[2] <= 0) dim[2] = 1;
    if (blkDim[0] <= 0) blkDim[0] = 1;
    if (blkDim[1] <= 0) blkDim[1] = 1;
    if (blkDim[2] <= 0) blkDim[2] = 1;

    mGridDim.x = iDivUp(dim[0], blkDim[0]);
    mGridDim.y = iDivUp(dim[1], blkDim[1]);
    mGridDim.z = iDivUp(dim[2], blkDim[2]);
    mBlockDim.x = iMin(dim[0], blkDim[0]);
    mBlockDim.y = iMin(dim[1], blkDim[1]);
    mBlockDim.z = iMin(dim[2], blkDim[2]);
  }

  template <class... Args>
  void operator()(Args &&... args)
  {
    mFunc<<<mGridDim, mBlockDim>>>(std::forward<Args>(args)...);
  }
};
