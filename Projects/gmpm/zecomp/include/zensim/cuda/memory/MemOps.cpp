#include "MemOps.hpp"

#include "zensim/cuda/Cuda.h"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    cudaMalloc(&ret, size);
    return ret;
  }

  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    cudaFree(ptr);
  }

  void memset(device_mem_tag, void *addr, int chval, std::size_t size) {
    cudaMemset(addr, chval, size);
  }
  void copy(device_mem_tag, void *dst, void *src, std::size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  }

  void *allocate(device_const_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    cudaMalloc(&ret, size);
    return ret;
  }
  void deallocate(device_const_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    cudaFree(ptr);
  }
  void memset(device_const_mem_tag, void *addr, int chval, std::size_t size) {
    cudaMemset(addr, chval, size);
  }
  void copy(device_const_mem_tag, void *dst, void *src, std::size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  }

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment) {
    void *ret{nullptr};
    cudaMallocManaged(&ret, size, cudaMemAttachGlobal);
    return ret;
  }
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment) {
    cudaFree(ptr);
  }
  void memset(um_mem_tag, void *addr, int chval, std::size_t size) {
    cudaMemset(addr, chval, size);
  }
  void copy(um_mem_tag, void *dst, void *src, std::size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  }
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did) {
    cudaMemoryAdvise option{};
    if (advice == "ACCESSED_BY")
      option = cudaMemAdviseSetAccessedBy;
    else if (advice == "PREFERRED_LOCATION")
      option = cudaMemAdviseSetPreferredLocation;
    else if (advice == "READ_MOSTLY")
      option = cudaMemAdviseSetReadMostly;
    else
      throw std::runtime_error(
          fmt::format("advise(tag um_mem_tag, advice {}, addr {}, bytes {}, devid {})\n", advice,
                      reinterpret_cast<std::uintptr_t>(addr), bytes, (int)did));
    if (Cuda::context(did).supportConcurrentUmAccess)
      cudaMemAdvise(addr, bytes, option, (int)did);
  }

}  // namespace zs