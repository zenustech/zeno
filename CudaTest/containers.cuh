#pragma once

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
      mData[i] = T();
    }
  }

  __host__ __device__ T &operator[](ssize_t i) {
    return mData[i];
  }

  __host__ __device__ T const &operator[](ssize_t i) const {
    return mData[i];
  }
};
