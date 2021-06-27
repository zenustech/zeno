#pragma once

#include <atomic>
#include <stdexcept>

#include "zensim/Reflection.h"
#include "zensim/Singleton.h"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/memory/Allocator.h"
#include "zensim/memory/MemOps.hpp"
#include "zensim/memory/MemoryResource.h"

namespace zs {

  struct GeneralAllocator {
    [[nodiscard]] void *allocate(std::size_t bytes,
                                 std::size_t alignment = alignof(std::max_align_t)) {
      return res->allocate(bytes, alignment);
    }
    void deallocate(void *p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) {
      res->deallocate(p, bytes, alignment);
    }
    bool is_equal(const mr_t &other) const noexcept { return res.get() == &other; }
    bool is_equal(const GeneralAllocator &other) const noexcept {
      return res.get() == other.res.get();
    }
    template <template <typename Tag> class AllocatorT, typename MemTag, typename... Args>
    void construct(MemTag, Args &&...args) {
      res = std::make_unique<AllocatorT<MemTag>>(FWD(args)...);
    }
    /// specifically for cuda uvm advise
    template <template <typename Tag> class AllocatorT>
    void construct(mem_tags tag, std::string_view advice, ProcID did) {
      match([this, advice, did](auto &tag) {
        res = std::make_unique<AllocatorT<remove_cvref_t<decltype(tag)>>>(advice, did);
      })(tag);
    }

    Holder<mr_t> res{};
  };

  /// global free function
  void record_allocation(mem_tags, void *, std::string_view, std::size_t = 0, std::size_t = 0);
  void erase_allocation(void *);
  void copy(MemoryEntity dst, MemoryEntity src, std::size_t size);

  GeneralAllocator get_memory_source(memsrc_e mre, ProcID devid,
                                     std::string_view advice = std::string_view{});

  struct Resource : Singleton<Resource> {
    static std::atomic_ullong &counter() noexcept { return instance()._counter; }

    struct AllocationRecord {
      mem_tags tag{};
      std::size_t size{0}, alignment{0};
      std::string allocatorType{};
#if 0
      GeneralAllocator getRawAllocator() const {
        GeneralAllocator ret{};
        match([&ret](auto &tag) { ret.construct<raw_allocator>(tag); })(tag);
        return ret;
      }
#endif
    };
    Resource() = default;
    ~Resource();

    void record(mem_tags tag, void *ptr, std::string_view name, std::size_t size,
                std::size_t alignment);
    void erase(void *ptr);

    void deallocate(void *ptr);

  private:
    mutable std::atomic_ullong _counter{0};
  };

  Resource &get_resource_manager() noexcept;

}  // namespace zs
