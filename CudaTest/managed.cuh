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
class _Traits {
};


class _IAccessor {
public:
  __host__ void activate() {
  }

  __host__ void deactivate() {
  }

  __host__ __device__ bool isActive() {
    return true;
  }
};


template <class PrimitiveT>
class Place : public Managed {
  PrimitiveT mValue;

public:
  class AccessorType : public _IAccessor {
    PrimitiveT &mValue;

  public:
    __host__ __device__ AccessorType(PrimitiveT &value) : mValue(value) {}

    __host__ __device__ PrimitiveT *get() {
      return &mValue;
    }
  };

  __host__ __device__ AccessorType access(size_t index = 0) {
    return mValue;
  }
};

template <class ValueT>
class _Traits<Place<ValueT>> {
public:
  static constexpr size_t Size = 1;
};


template <class ValueT, size_t SizeT>
class Dense : public Managed {
  ValueT mData[SizeT];

public:
  class AccessorType : public _IAccessor {
    ValueT &mValue;

  public:
    __host__ __device__ AccessorType(ValueT &value) : mValue(value) {}

    __host__ __device__ ValueT *get() {
      return &mValue;
    }
  };

  __host__ __device__ AccessorType access(size_t index) {
    return mData[index];
  }
};

template <class ValueT, size_t SizeT>
class _Traits<Dense<ValueT, SizeT>> {
public:
  static constexpr size_t Size = SizeT;
  using ValueType = ValueT;
};


template <class ValueT>
class Pointer : public Managed {
  ValueT *mPtr;

public:
  class AccessorType : public _IAccessor {
    ValueT *&mPtr;

  public:
    __host__ __device__ AccessorType(ValueT *&ptr) : mPtr(ptr) {}

    __host__ __device__ ValueT *get() {
      return mPtr;
    }

    __host__ void activate() {
      if (!mPtr) {
        mPtr = new ValueT();
      }
    }

    __host__ void deactivate() {
      if (mPtr) {
        delete mPtr;
        mPtr = nullptr;
      }
    }

    __host__ __device__ bool isActive() {
      if (mPtr) {
        return true;
      } else {
        return false;
      }
    }
  };

  __host__ __device__ AccessorType access(size_t index = 0) {
    return mPtr;
  }
};

template <class ValueT>
class _Traits<Pointer<ValueT>> {
public:
  static constexpr size_t Size = 1;
  using ValueType = ValueT;
};


template <class ContainerT>
class Subscriptor : public _IAccessor
{
  using ChunkType = typename _Traits<ContainerT>::ValueType;
  using ContainerAccessorType = typename ContainerT::AccessorType;
  using ChunkAccessorType = typename ChunkType::AccessorType;
  static constexpr size_t ChunkSize = _Traits<ChunkType>::Size;

  ContainerT &container;
  size_t chunkIndex;
  size_t chunkOffset;

  __host__ __device__ ContainerAccessorType getChunkAccessor() {
    return container.access(chunkIndex);
  }

public:
  __host__ __device__ Subscriptor(
      ContainerT &container, size_t index) : container(container) {
    chunkIndex = index / ChunkSize;
    chunkOffset = index % ChunkSize;
  }

  __host__ __device__ auto get() -> decltype(auto) {
    auto chunkAccessor = getChunkAccessor();
    assert(chunkAccessor.isActive());
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, chunkOffset);
    return chunkSubscriptor.get();
  }

  __host__ __device__ bool isActive() {
    auto chunkAccessor = getChunkAccessor();
    if (!chunkAccessor.isActive())
      return false;
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, chunkOffset);
    return chunkSubscriptor.isActive();
  }

  __host__ void activate() {
    auto chunkAccessor = getChunkAccessor();
    chunkAccessor.activate();
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, chunkOffset);
    chunkSubscriptor.activate();
  }

  __host__ void deactivate() {
    auto chunkAccessor = getChunkAccessor();
    if (!chunkAccessor.isActive())
      return;
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, chunkOffset);
    chunkSubscriptor.deactivate();
  }
};

template <class PrimitiveT>
class Subscriptor<Place<PrimitiveT>> : public _IAccessor
{
  using PlaceAccessorType = typename Place<PrimitiveT>::AccessorType;
  PlaceAccessorType mAccessor;

public:
  __host__ __device__ Subscriptor(
      Place<PrimitiveT> &container, size_t index)
    : mAccessor(container.access(index)) {
  }

  __host__ __device__ PrimitiveT *get() {
    return mAccessor.get();
  }
};
