#pragma once

#include "helper_cuda.h"
#include <cassert>
#include <memory>
#include <array>


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


using dim_t = size_t;


class Indices {
  size_t mIndices[8];

public:
  __host__ __device__ Indices(size_t index = 0) : Indices{index} {
  }

  __host__ __device__ Indices(std::initializer_list<size_t> indices) {
    size_t i = 0;
    for (auto const &index: indices) {
      mIndices[i++] = index;
    }
    for (; i < 8; i++) {
      mIndices[i] = 0;
    }
  }

  __host__ __device__ size_t &operator[](dim_t dim) {
    return mIndices[dim];
  }

  __host__ __device__ size_t const &operator[](dim_t dim) const {
    return mIndices[dim];
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

  __host__ __device__ AccessorType access(Indices const &index = 0) {
    return mValue;
  }
};

template <class ValueT>
class _Traits<Place<ValueT>> {
public:
  static constexpr size_t Size = 1;
  static constexpr dim_t Dim = 8;
};


template <class ValueT, size_t SizeT, dim_t DimT>
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

  __host__ __device__ AccessorType access(Indices const &index) {
    return mData[index[DimT]];
  }
};

template <class ValueT, size_t SizeT, dim_t DimT>
class _Traits<Dense<ValueT, SizeT, DimT>> {
public:
  static constexpr size_t Size = SizeT;
  static constexpr dim_t Dim = DimT;
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

  __host__ __device__ AccessorType access(Indices const &index = 0) {
    return mPtr;
  }
};

template <class ValueT>
class _Traits<Pointer<ValueT>> {
public:
  static constexpr size_t Size = 1;
  static constexpr dim_t Dim = 8;
  using ValueType = ValueT;
};


template <class ContainerT>
class Subscriptor : public _IAccessor
{
  using ChunkType = typename _Traits<ContainerT>::ValueType;
  using ContainerAccessorType = typename ContainerT::AccessorType;
  using ChunkAccessorType = typename ChunkType::AccessorType;
  static constexpr size_t ChunkSize = _Traits<ChunkType>::Size;
  static constexpr size_t ChunkDim = _Traits<ChunkType>::Dim;

  ContainerT &mContainer;
  Indices mChunkIndex;
  Indices mChunkOffset;

  __host__ __device__ ContainerAccessorType getChunkAccessor() {
    return mContainer.access(mChunkIndex);
  }

public:
  __host__ __device__ Subscriptor(
      ContainerT &container, Indices const &index) : mContainer(container) {
    mChunkOffset = mChunkIndex = index;
    if constexpr (ChunkDim != 8) {
      mChunkIndex[ChunkDim] /= ChunkSize;
      mChunkOffset[ChunkDim] %= ChunkSize;
    }
  }

  __host__ __device__ auto get() -> decltype(auto) {
    auto chunkAccessor = getChunkAccessor();
    assert(chunkAccessor.isActive());
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, mChunkOffset);
    return chunkSubscriptor.get();
  }

  __host__ __device__ bool isActive() {
    auto chunkAccessor = getChunkAccessor();
    if (!chunkAccessor.isActive())
      return false;
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, mChunkOffset);
    return chunkSubscriptor.isActive();
  }

  __host__ void activate() {
    auto chunkAccessor = getChunkAccessor();
    chunkAccessor.activate();
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, mChunkOffset);
    chunkSubscriptor.activate();
  }

  __host__ void deactivate() {
    auto chunkAccessor = getChunkAccessor();
    if (!chunkAccessor.isActive())
      return;
    ChunkType &chunk = *chunkAccessor.get();
    Subscriptor<ChunkType> chunkSubscriptor(chunk, mChunkOffset);
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
      Place<PrimitiveT> &container, Indices const &index)
    : mAccessor(container.access(index)) {
  }

  __host__ __device__ PrimitiveT *get() {
    return mAccessor.get();
  }
};


template <class ContainerT>
class Field {
  std::unique_ptr<ContainerT> mPtr;

public:
  Field() : mPtr(std::make_unique<ContainerT>()) {
  }

  class Copy {
    ContainerT *mPtr;

    Copy(Field const &field) : mPtr(field.mPtr.get()) {
    }

    Subscriptor<ContainerT> subscript(Indices const &indices) {
      return Subscriptor<ContainerT>(*mPtr, indices);
    }
  };

  Copy copy() {
    return Copy(*this);
  }
};
