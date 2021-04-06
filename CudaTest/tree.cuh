#pragma once

#include "helper_cuda.h"
#include <cassert>


__host__ void cMalloc(size_t len) {
  void *ptr;
  checkCudaErrors(cudaMallocManaged(&ptr, len));
  return ptr;
}

__host__ void cFree(void *ptr) {
  checkCudaErrors(cudaFree(ptr));
}


class Dense {
  void *mData;

  size_t childSize;
  std::array<size_t, 8> childCount;
};
