#pragma once

#include <memory>
#include <type_traits>

#include "MemOps.hpp"
#include "MemoryResource.h"
#include "zensim/Singleton.h"
#include "zensim/math/bit/Bits.h"
#include "zensim/memory/MemOps.hpp"

namespace zs {

  extern void record_allocation(mem_tags, void *, std::string_view, std::size_t, std::size_t);
  extern void erase_allocation(void *);
  template <typename MemTag> struct raw_allocator : mr_t, Singleton<raw_allocator<MemTag>> {
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      if (bytes) {
        auto ret = zs::allocate(MemTag{}, bytes, alignment);
        record_allocation(MemTag{}, ret, demangle(*this), bytes, alignment);
        return ret;
      } else
        return nullptr;
    }
    void do_deallocate(void *ptr, std::size_t bytes, std::size_t alignment) override {
      zs::deallocate(MemTag{}, ptr, bytes, alignment);
      erase_allocation(ptr);
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }
  };

  template <typename MemTag> struct advisor_allocator : mr_t {
    advisor_allocator(std::string_view option = "PREFERRED_LOCATION", ProcID did = 0,
                      mr_t *up = &raw_allocator<MemTag>::instance())
        : upstream{up}, option{option}, did{did} {}
    ~advisor_allocator() = default;
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      void *ret = upstream->allocate(bytes, alignment);
      advise(MemTag{}, option, ret, bytes, did);
      return ret;
    }
    void do_deallocate(void *ptr, std::size_t bytes, std::size_t alignment) override {
      upstream->deallocate(ptr, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override { return this == &other; }

  private:
    mr_t *upstream;
    std::string option;
    ProcID did;
  };

  class handle_resource : mr_t {
  public:
    explicit handle_resource(mr_t *upstream) noexcept;
    handle_resource(std::size_t initSize, mr_t *upstream) noexcept;
    handle_resource() noexcept;
    ~handle_resource() override;

    mr_t *upstream_resource() const noexcept { return _upstream; }

    void *handle() const noexcept { return _handle; }
    void *address(uintptr_t offset) const noexcept { return (_handle + offset); }
    uintptr_t acquire(std::size_t bytes, std::size_t alignment) {
      char *ret = (char *)this->do_allocate(bytes, alignment);
      return ret - _handle;
    }

  protected:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const mr_t &other) const noexcept override;

  private:
    std::size_t _bufSize{128 * sizeof(void *)}, _align;
    mr_t *const _upstream{nullptr};
    char *_handle{nullptr}, *_head{nullptr};
  };

  /// https://en.cppreference.com/w/cpp/named_req/Allocator#Allocator_completeness_requirements
  // An allocator type X for type T additionally satisfies the allocator
  // completeness requirements if both of the following are true regardless of
  // whether T is a complete type: X is a complete type Except for value_type, all
  // the member types of std::allocator_traits<X> are complete types.

#if 0
  /// for automatic dynamic memory management
  struct memory_pools : Singleton<memory_pools>, mr_t {
    /// https://stackoverflow.com/questions/46509152/why-in-x86-64-the-virtual-address-are-4-bits-shorter-than-physical-48-bits-vs

    using poolid = unsigned char;
    static constexpr poolid nPools = 4;
    /// 9-bit per page-level: 512, 4K, 2M, 1G
    static constexpr std::size_t block_bits[nPools] = {9, 12, 21, 30};
    static constexpr std::size_t block_sizes(poolid pid) noexcept {
      return static_cast<std::size_t>(1) << block_bits[pid];
    }
    static constexpr poolid pool_index(std::size_t bytes) noexcept {
      const poolid nbits = bit_count(bytes);
      for (poolid i = 0; i < nPools; ++i)
        if (block_bits[i] > nbits) return i;
      return nPools - 1;
    }

    memory_pools(mr_t *source) {
      for (char i = 0; i < nPools; ++i) {
        pmr::pool_options opt{/*.max_blocks_per_chunk = */ 0,
                              /*.largest_required_pool_block = */ block_sizes(i)};
        /// thread-safe version
        _pools[i] = std::make_unique<synchronized_pool_resource>(opt, source);
      }
    }

  protected:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      const poolid pid = pool_index(bytes);
      return _pools[pid]->allocate(bytes, alignment);
    }

    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override {
      const poolid pid = pool_index(bytes);
      _pools[pid]->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override {
      return this == dynamic_cast<memory_pools *>(const_cast<mr_t *>(&other));
    }

  private:
    std::array<std::unique_ptr<mr_t>, nPools> _pools;
  };

  template <std::size_t... Ns> struct static_memory_pools : mr_t {
    using poolid = char;
    static constexpr poolid nPools = sizeof...(Ns);
    static constexpr std::size_t block_bits[nPools] = {Ns...};
    static constexpr std::size_t block_sizes(poolid pid) noexcept {
      return static_cast<std::size_t>(1) << block_bits[pid];
    }
    static constexpr poolid pool_index(std::size_t bytes) noexcept {
      const poolid nbits = bit_count(bytes);
      for (poolid i = 0; i < nPools; ++i)
        if (block_bits[i] > nbits) return i;
      return nPools - 1;
    }

    static_memory_pools(mr_t *source) {
      for (char i = 0; i < nPools; ++i) {
        pmr::pool_options opt{/*.max_blocks_per_chunk = */ 0,
                              /*.largest_required_pool_block = */ block_sizes(i)};
        _pools[i] = std::make_unique<unsynchronized_pool_resource>(opt, source);
      }
    }

  protected:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      const poolid pid = pool_index(bytes);
      return _pools[pid]->allocate(bytes, alignment);
    }

    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override {
      const poolid pid = pool_index(bytes);
      _pools[pid]->deallocate(p, bytes, alignment);
    }
    bool do_is_equal(const mr_t &other) const noexcept override {
      return this == dynamic_cast<static_memory_pools *>(const_cast<mr_t *>(&other));
    }

  private:
    std::array<std::unique_ptr<mr_t>, nPools> _pools;
  };
#endif

  struct general_allocator {
    general_allocator() noexcept : _mr{&raw_allocator<host_mem_tag>::instance()} {};
    general_allocator(const general_allocator &other) : _mr{other.resource()} {}
    general_allocator(mr_t *r) noexcept : _mr{r} {}

    mr_t *resource() const { return _mr; }

    void *allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
      return resource()->allocate(bytes, align);
    }
    void deallocate(void *p, std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
      resource()->deallocate(p, bytes, align);
    }

  private:
    mr_t *_mr{nullptr};
  };

  struct heap_allocator : general_allocator {
    heap_allocator() : general_allocator{&raw_allocator<host_mem_tag>::instance()} {}
  };

  struct stack_allocator {
    explicit stack_allocator(mr_t *mr, std::size_t totalMemBytes, std::size_t alignBytes);
    stack_allocator() = delete;
    ~stack_allocator();

    mr_t *resource() const noexcept { return _mr; }

    /// from taichi
    void *allocate(std::size_t bytes);
    void deallocate(void *p, std::size_t);
    void reset() { _head = _data; }

    char *_data, *_head, *_tail;
    std::size_t _align;

  private:
    mr_t *_mr{nullptr};
  };

}  // namespace zs
