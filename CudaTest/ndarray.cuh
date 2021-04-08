#pragma once

#include "helper_cuda.h"
#include <utility>

template <class T, size_t N>
class Array {
  T mData[N];

public:
  Array() = default;
  Array(Array const &) = default;
  Array(Array &&) = default;

  __host__ __device__ Array(std::initializer_list<T> const &args) {
    int i = 0;
    for (auto const &value: args) {
      mData[i++] = value;
    }
    for (; i < N; i++) {
      mData[i] = T(0);
    }
  }

  __host__ __device__ T &operator[](ssize_t i) {
    return mData[i];
  }

  __host__ __device__ T const &operator[](ssize_t i) const {
    return mData[i];
  }
};

static constexpr size_t NDims = 4;

class NDView {
protected:
  Array<ssize_t, NDims> mStride;
  Array<ssize_t, NDims> mShape;
  void *mBase{nullptr};

  NDView() = default;

public:
  NDView(NDView const &) = default;
  NDView(NDView &&) = default;

  __host__ __device__ NDView(
      Array<ssize_t, NDims> const &stride,
      Array<ssize_t, NDims> const &shape,
      void *base = nullptr)
    : mBase(base)
    , mStride(stride)
    , mShape(shape)
  {}

  __host__ __device__ Array<ssize_t, NDims> const &stride() const {
    return mStride;
  }

  __host__ __device__ Array<ssize_t, NDims> const &shape() const {
    return mShape;
  }

  __host__ __device__ void *data() const {
    return mBase;
  }

  __host__ __device__ ssize_t size() const {
    ssize_t offset = 0;
    for (int i = 0; i < NDims; i++) {
      offset += mStride[i] * mShape[i];
    }
    return offset;
  }

  __host__ __device__ void *operator()(Array<ssize_t, NDims> const &indices) const {
    ssize_t offset = 0;
    for (int i = 0; i < NDims; i++) {
      offset += mStride[i] * indices[i];
    }
    return (char *)mBase + offset;
  }
};

class NDArray : public NDView {
public:
  __host__ NDArray(
      Array<ssize_t, NDims> const &stride,
      Array<ssize_t, NDims> const &shape)
    : NDView(stride, shape)
  {
    checkCudaErrors(cudaMallocManaged(&mBase, this->size()));
  }

  __host__ ~NDArray() {
    checkCudaErrors(cudaFree(mBase));
    mBase = nullptr;
  }
};

template <class T>
class NDTypedView : public NDView {
public:
  __host__ __device__ T &operator()(Array<ssize_t, NDims> const &indices) const {
    void *ptr = NDView::operator()(indices);
    return *static_cast<T *>(ptr);
  }

  NDTypedView(NDView const &view) : NDView(view) {}
  NDTypedView(NDView &&view) : NDView(view) {}
};

__global__ void blur(NDTypedView<int> arr)
{
  ssize_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  arr({ix}) = ix + 1;
}
