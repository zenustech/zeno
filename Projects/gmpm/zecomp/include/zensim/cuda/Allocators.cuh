#pragma once
#include "zensim/memory/Allocator.h"

namespace zs {

  struct MonotonicAllocator : stack_allocator {
    MonotonicAllocator(std::size_t totalMemBytes, std::size_t alignment);
    auto borrow(std::size_t bytes) -> void *;
    void reset();
  };
  struct MonotonicVirtualAllocator : stack_allocator {
    MonotonicVirtualAllocator(int devid, std::size_t totalMemBytes, std::size_t alignment);
    auto borrow(std::size_t bytes) -> void *;
    void reset();
  };

}  // namespace zs
