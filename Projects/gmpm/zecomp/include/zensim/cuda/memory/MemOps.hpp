#pragma once
#include "zensim/memory/MemoryResource.h"

namespace zs {

  void *allocate(device_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(device_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(device_mem_tag, void *addr, int chval, std::size_t size);
  void copy(device_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(device_const_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(device_const_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(device_const_mem_tag, void *addr, int chval, std::size_t size);
  void copy(device_const_mem_tag, void *dst, void *src, std::size_t size);

  void *allocate(um_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(um_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(um_mem_tag, void *addr, int chval, std::size_t size);
  void copy(um_mem_tag, void *dst, void *src, std::size_t size);
  void advise(um_mem_tag, std::string advice, void *addr, std::size_t bytes, ProcID did);

}  // namespace zs