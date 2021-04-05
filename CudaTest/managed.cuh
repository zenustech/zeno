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

template <typename T>
class ManagedSlot : public Managed {
  T core;

public:
  __host__ __device__ T &get() {
    return core;
  }

  __host__ __device__ T const &get() const {
    return core;
  }
};

template <typename T, size_t N>
class ManagedArray : public Managed {
  T core[N];

public:
  __host__ __device__ T &operator()(int i) {
    return core[i];
  }

  __host__ __device__ T const &operator()(int i) const {
    return core[i];
  }
};

class Memory : public Managed {
  void *ptr{nullptr};
  size_t len{0};

  __host__ __device__ Memory(Memory const &) = delete;
  __host__ __device__ Memory(Memory &&) = delete;

public:
  __host__ Memory(size_t len) : len(len) {
    printf("Memory(%d)\n", (int)len);
    cudaMallocManaged(&ptr, len);
  }

  __host__ __device__ void *data() const {
    return ptr;
  }

  __host__ __device__ size_t size() const {
    return len;
  }

  __host__ ~Memory() {
    printf("~Memory(%d)\n", (int)len);
    cudaFree(ptr);
  }
};

template <typename T>
class MemoryArray : public Managed {
  Memory mem;
  size_t len;

  __host__ __device__ MemoryArray(MemoryArray const &) = delete;
  __host__ __device__ MemoryArray(MemoryArray &&) = delete;

public:
  __host__ MemoryArray(size_t len)
    : len(len), mem(len * sizeof(T)) {
  }

  __host__ __device__ T *data() const {
    return (T *)mem.data();
  }

  __host__ __device__ size_t size() const {
    return len;
  }

  __host__ __device__ T &operator()(int i) const {
    return data()[i];
  }
};
