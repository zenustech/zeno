#pragma once
#include "zensim/memory/MemoryResource.h"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/memory/MemOps.hpp"
#endif

namespace zs {

  void *allocate(host_mem_tag, std::size_t size, std::size_t alignment);
  void deallocate(host_mem_tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset(host_mem_tag, void *addr, int chval, std::size_t size);
  void copy(host_mem_tag, void *dst, void *src, std::size_t size);

#if 0
  /// dispatch mem op calls
  void *allocate_dispatch(mem_tags tag, std::size_t size, std::size_t alignment);
  void deallocate_dispatch(mem_tags tag, void *ptr, std::size_t size, std::size_t alignment);
  void memset_dispatch(mem_tags tag, void *addr, int chval, std::size_t size);
  void copy_dispatch(mem_tags tag, void *dst, void *src, std::size_t size);
  void advise_dispatch(mem_tags tag, std::string advice, void *addr, std::size_t bytes, ProcID did);
#endif

  /// default memory operation implementations (fallback)
  template <typename MemTag> void *allocate(MemTag, std::size_t size, std::size_t alignment) {
    throw std::runtime_error(
        fmt::format("allocate(tag {}, size {}, alignment {}) not implemented\n",
                    get_memory_source_tag(MemTag{}), size, alignment));
  }
  template <typename MemTag>
  void deallocate(MemTag, void *ptr, std::size_t size, std::size_t alignment) {
    throw std::runtime_error(fmt::format(
        "deallocate(tag {}, ptr {}, size {}, alignment {}) not implemented\n",
        get_memory_source_tag(MemTag{}), reinterpret_cast<std::uintptr_t>(ptr), size, alignment));
  }
  template <typename MemTag> void memset(MemTag, void *addr, int chval, std::size_t size) {
    throw std::runtime_error(fmt::format(
        "memset(tag {}, ptr {}, charval {}, size {}) not implemented\n",
        get_memory_source_tag(MemTag{}), reinterpret_cast<std::uintptr_t>(addr), chval, size));
  }
  template <typename MemTag> void copy(MemTag, void *dst, void *src, std::size_t size) {
    throw std::runtime_error(fmt::format(
        "copy(tag {}, dst {}, src {}, size {}) not implemented\n", get_memory_source_tag(MemTag{}),
        reinterpret_cast<std::uintptr_t>(dst), reinterpret_cast<std::uintptr_t>(src), size));
  }
  template <typename MemTag, typename... Args>
  void advise(MemTag, std::string advice, void *addr, Args...) {
    throw std::runtime_error(
        fmt::format("advise(tag {}, advise {}, addr {}) with {} args not implemented\n",
                    get_memory_source_tag(MemTag{}), advice, reinterpret_cast<std::uintptr_t>(addr),
                    sizeof...(Args)));
  }

}  // namespace zs