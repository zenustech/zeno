#pragma once

#include "helper_cuda.h"
#include <cassert>

class Managed {
public:
  __host__ void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    return ptr;
  }

  __host__ void operator delete(void *ptr) {
    cudaFree(ptr);
  }
};

template <class ValueT>
class Place : public Managed {
  ValueT mValue;

public:
  __host__ __device__ ValueT &get() {
    return mValue;
  }
};

template <class ValueT, size_t SizeT>
class Dense : public Managed {
  ValueT mData[SizeT];

public:
  __host__ __device__ ValueT &get(size_t i) {
    return mData[i];
  }
};

template <class ValueT>
class Pointer : public Managed {
  ValueT *mPtr;

public:
  __host__ __device__ ValueT &get() {
    return *mPtr;
  }

  __host__ __device__ void activate() {
    if (!mPtr) {
      mPtr = new ValueT();
    }
  }

  __host__ __device__ void deactivate() {
    if (mPtr) {
      delete mPtr;
      mPtr = nullptr;
    }
  }

  __host__ __device__ bool is_active() {
    if (mPtr) {
      return true;
    } else {
      return false;
    }
  }
};
