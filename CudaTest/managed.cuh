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
class Traits {
};


template <class ValueT>
class Place : public Managed {
  ValueT mValue;

public:
  __host__ __device__ ValueT &get(size_t index = 0) {
    return mValue;
  }
};

template <class ValueT>
class Traits<Place<ValueT>> {
public:
  static constexpr size_t Size = 1;
  using ValueType = void;
};


template <class ValueT, size_t SizeT>
class Dense : public Managed {
  ValueT mData[SizeT];

public:
  __host__ __device__ ValueT &get(size_t index) {
    return mData[index];
  }
};

template <class ValueT, size_t SizeT>
class Traits<Dense<ValueT, SizeT>> {
public:
  static constexpr size_t Size = SizeT;
  using ValueType = ValueT;
};


template <class ValueT>
class Pointer : public Managed {
  ValueT *mPtr;

public:
  __host__ __device__ ValueT &get(size_t index = 0) {
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

template <class ValueT>
class Traits<Pointer<ValueT>> {
public:
  static constexpr size_t Size = 1;
  using ValueType = ValueT;
};


template <class PrimitiveT, class ContainerT>
__host__ __device__ PrimitiveT &subscript(
    ContainerT &container, size_t index) {
  using ChunkType = typename Traits<ContainerT>::ValueType;
  constexpr size_t ChunkSize = Traits<ChunkType>::Size;
  size_t chunkIndex = index / ChunkSize;
  size_t chunkOffset = index % ChunkSize;
  ChunkType &chunk = container.get(chunkIndex);
  return subscript(chunk, chunkOffset);
}

template <class PrimitiveT>
__host__ __device__ PrimitiveT &subscript(
    Place<PrimitiveT> &container, size_t index) {
  return container.get(index);
}
