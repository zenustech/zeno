#pragma once

#include "helper_cuda.h"
#include <cassert>


class Managed {
public:
  __host__ void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  __host__ void operator delete(void *ptr) {
    checkCudaErrors(cudaFree(ptr));
  }
};


class TreeNode {
  int *mData;

  __host__ __device__ TreeNode() {
  }
};
