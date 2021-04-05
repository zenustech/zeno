#pragma once

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

template <typename T, size_t N>
class Array : public Managed {
  T core[N];

public:
  __host__ __device__ T &operator()(int i) {
    return core[i];
  }

  __host__ __device__ T const &operator()(int i) const {
    return core[i];
  }
};
