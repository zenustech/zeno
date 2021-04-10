#pragma once

#include "helper_cuda.h"
#include "Array.cuh"

static constexpr size_t NAxes = 4;

class NDView {
protected:
  void *mBase{nullptr};
  Array<size_t, NAxes> mStride;
  Array<size_t, NAxes> mShape;

  NDView() = default;

public:
  NDView(NDView const &) = default;
  NDView(NDView &&) = default;

  __host__ __device__ NDView(
      Array<size_t, NAxes> const &shape,
      Array<size_t, NAxes> const &stride,
      void *base = nullptr)
    : mBase(base)
    , mStride(stride)
    , mShape(shape)
  {}

  __host__ __device__ NDView(
      Array<size_t, NAxes> const &shape,
      size_t elementSize,
      void *base = nullptr)
    : mBase(base)
    , mShape(shape)
  {
    size_t term = elementSize;
    for (int i = 0; i < NAxes; i++) {
      mStride[i] = term;
      term *= mShape[i];
    }
  }

  __host__ __device__ Array<size_t, NAxes> const &stride() const {
    return mStride;
  }

  __host__ __device__ Array<size_t, NAxes> &stride() {
    return mStride;
  }

  __host__ __device__ Array<size_t, NAxes> const &shape() const {
    return mShape;
  }

  __host__ __device__ Array<size_t, NAxes> &shape() {
    return mShape;
  }

  __host__ __device__ void *const &data() const {
    return mBase;
  }

  __host__ __device__ void *&data() {
    return mBase;
  }

  __host__ __device__ size_t size() const {
    size_t offset = 0;
    for (int i = 0; i < NAxes; i++) {
      offset += mStride[i] * mShape[i];
    }
    return offset;
  }

  __host__ __device__ void *operator()(
      Array<size_t, NAxes> const &indices) const {
    size_t offset = 0;
    for (int i = 0; i < NAxes; i++) {
      offset += mStride[i] * indices[i];
    }
    return (char *)mBase + offset;
  }
};

class NDArray : public NDView {
  void do_allocate() {
    checkCudaErrors(cudaMallocManaged((void **)&mBase, this->size()));
  }

  void do_release() {
    checkCudaErrors(cudaFree(mBase));
    mBase = nullptr;
  }

public:
  __host__ NDArray(
      Array<size_t, NAxes> const &shape,
      Array<size_t, NAxes> const &stride)
    : NDView(shape, stride)
  {
    do_allocate();
  }

  __host__ NDArray(
      Array<size_t, NAxes> const &shape,
      size_t elementSize)
    : NDView(shape, elementSize)
  {
    do_allocate();
  }

  __host__ ~NDArray() {
    do_release();
  }
};

template <class T>
class NDTypedArray : public NDArray {
public:
  __host__ NDTypedArray(
      Array<size_t, NAxes> const &shape,
      Array<size_t, NAxes> const &stride)
    : NDArray(shape, stride)
  {
  }

  __host__ NDTypedArray(
      Array<size_t, NAxes> const &shape,
      size_t elementSize = sizeof(T))
    : NDArray(shape, elementSize)
  {
  }

  __host__ T &operator()(Array<size_t, NAxes> const &indices) const {
    void *ptr = NDArray::operator()(indices);
    return *static_cast<T *>(ptr);
  }
};

template <class T>
class NDTypedView : public NDView {
public:
  __host__ __device__ T &operator()(Array<size_t, NAxes> const &indices) const {
    void *ptr = NDView::operator()(indices);
    return *static_cast<T *>(ptr);
  }

  __host__ __device__ NDTypedView(NDView const &view) : NDView(view) {}
  __host__ __device__ NDTypedView(NDView &&view) : NDView(view) {}

  __host__ __device__ NDTypedView(
      Array<size_t, NAxes> const &shape,
      Array<size_t, NAxes> const &stride,
      void *base = nullptr)
    : NDView(shape, stride, base) {}

  __host__ __device__ NDTypedView(
      Array<size_t, NAxes> const &shape,
      void *base = nullptr)
    : NDView(shape, sizeof(T), base) {}

  __host__ __device__ NDTypedView(
      Array<size_t, NAxes> const &shape,
      size_t elementSize,
      void *base = nullptr)
    : NDView(shape, elementSize, base) {}
};
