#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "zensim/tpls/fmt/format.h"
#include "zensim/types/Function.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  // namespace pmr = std::pmr;

  /// since we cannot use memory_resource header in libstdc++
  /// we directly use its implementation
  class memory_resource {
    static constexpr size_t _S_max_align = alignof(max_align_t);

  public:
    memory_resource() = default;
    memory_resource(const memory_resource&) = default;
    virtual ~memory_resource() = default;  // key function

    memory_resource& operator=(const memory_resource&) = default;

    [[nodiscard]] void* allocate(size_t __bytes, size_t __alignment = _S_max_align)
        __attribute__((__returns_nonnull__, __alloc_size__(2), __alloc_align__(3))) {
      return do_allocate(__bytes, __alignment);
    }

    void deallocate(void* __p, size_t __bytes, size_t __alignment = _S_max_align)
        __attribute__((__nonnull__)) {
      return do_deallocate(__p, __bytes, __alignment);
    }

    bool is_equal(const memory_resource& __other) const noexcept { return do_is_equal(__other); }

  private:
    virtual void* do_allocate(size_t __bytes, size_t __alignment) = 0;

    virtual void do_deallocate(void* __p, size_t __bytes, size_t __alignment) = 0;

    virtual bool do_is_equal(const memory_resource& __other) const noexcept = 0;
  };

  inline bool operator==(const memory_resource& __a, const memory_resource& __b) noexcept {
    return &__a == &__b || __a.is_equal(__b);
  }

#if __cpp_impl_three_way_comparison < 201907L
  inline bool operator!=(const memory_resource& __a, const memory_resource& __b) noexcept {
    return !(__a == __b);
  }
#endif

  using mr_t = memory_resource;
  // using unsynchronized_pool_resource = pmr::unsynchronized_pool_resource;
  // using synchronized_pool_resource = pmr::synchronized_pool_resource;
  // template <typename T> using object_allocator = pmr::polymorphic_allocator<T>;

  // HOST, DEVICE, DEVICE_CONST, UM, PINNED, FILE
  enum struct memsrc_e : unsigned char { host = 0, device, device_const, um, pinned, file };

  using host_mem_tag = wrapv<memsrc_e::host>;
  using device_mem_tag = wrapv<memsrc_e::device>;
  using device_const_mem_tag = wrapv<memsrc_e::device_const>;
  using um_mem_tag = wrapv<memsrc_e::um>;
  using pinned_mem_tag = wrapv<memsrc_e::pinned>;
  using file_mem_tag = wrapv<memsrc_e::file>;

  using mem_tags = variant<host_mem_tag, device_mem_tag, device_const_mem_tag, um_mem_tag,
                           pinned_mem_tag, file_mem_tag>;

  constexpr host_mem_tag mem_host{};
  constexpr device_mem_tag mem_device{};
  constexpr device_const_mem_tag mem_device_const{};
  constexpr um_mem_tag mem_um{};
  constexpr pinned_mem_tag mem_pinned{};
  constexpr file_mem_tag mem_file{};

  constexpr mem_tags to_memory_source_tag(memsrc_e loc) {
    mem_tags ret{};
    switch (loc) {
      case memsrc_e::host:
        ret = mem_host;
        break;
      case memsrc_e::device:
        ret = mem_device;
        break;
      case memsrc_e::device_const:
        ret = mem_device_const;
        break;
      case memsrc_e::um:
        ret = mem_um;
        break;
      case memsrc_e::pinned:
        ret = mem_pinned;
        break;
      case memsrc_e::file:
        ret = mem_file;
        break;
      default:;
    }
    return ret;
  }

  constexpr const char* memory_source_tag[]
      = {"HOST", "DEVICE", "DEVICE_CONST", "UM", "PINNED", "FILE"};
  constexpr const char* get_memory_source_tag(memsrc_e loc) {
    return memory_source_tag[static_cast<unsigned char>(loc)];
  }

  struct MemoryHandle {
    constexpr ProcID devid() const noexcept { return _devid; }
    constexpr memsrc_e memspace() const noexcept { return _memsrc; }
    constexpr MemoryHandle memoryHandle() const noexcept {
      return static_cast<MemoryHandle>(*this);
    }

    void swap(MemoryHandle& o) noexcept {
      std::swap(_devid, o._devid);
      std::swap(_memsrc, o._memsrc);
    }

    constexpr bool onHost() const noexcept { return _memsrc == memsrc_e::host; }
    constexpr const char* memSpaceName() const { return get_memory_source_tag(memspace()); }
    constexpr mem_tags getTag() const { return to_memory_source_tag(_memsrc); }

    memsrc_e _memsrc{memsrc_e::host};  // memory source
    ProcID _devid{-1};                 // cpu id
  };

  struct MemoryEntity {
    MemoryHandle descr{};
    void* ptr{nullptr};
    MemoryEntity() = default;
    template <typename T> constexpr MemoryEntity(MemoryHandle mh, T&& ptr)
        : descr{mh}, ptr{(void*)ptr} {}
  };

  // host = 0, device, device_const, um, pinned file
  constexpr mem_tags memop_tag(const MemoryHandle a, const MemoryHandle b) {
    auto spaceA = static_cast<unsigned char>(a._memsrc);
    auto spaceB = static_cast<unsigned char>(b._memsrc);
    if (spaceA > spaceB) std::swap(spaceA, spaceB);
    if (a._memsrc == b._memsrc) return to_memory_source_tag(a._memsrc);
    /// avoid um issue
    else if (spaceB < static_cast<unsigned char>(memsrc_e::um))
      return to_memory_source_tag(memsrc_e::device);
    else if (spaceB == static_cast<unsigned char>(memsrc_e::um))
      return to_memory_source_tag(memsrc_e::um);
    else
      throw std::runtime_error(fmt::format("memop_tag for ({}, {}) is undefined!",
                                           get_memory_source_tag(a._memsrc),
                                           get_memory_source_tag(b._memsrc)));
  }

}  // namespace zs
