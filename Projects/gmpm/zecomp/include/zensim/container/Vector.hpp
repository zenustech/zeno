#pragma once
#include <type_traits>

#include "zensim/memory/Allocator.h"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"
#include "zensim/types/Iterator.h"
#include "zensim/types/RuntimeStructurals.hpp"

namespace zs {

  template <typename Snode, typename Index = std::size_t> using vector_snode
      = ds::snode_t<ds::static_decorator<>, ds::uniform_domain<0, Index, 1, index_seq<0>>,
                    tuple<Snode>, vseq_t<1>>;
  template <typename T, typename Index = std::size_t> using vector_instance
      = ds::instance_t<ds::dense, vector_snode<wrapt<T>, Index>>;

  template <typename T, typename Index = std::size_t> struct Vector : vector_instance<T, Index>,
                                                                      MemoryHandle {
    /// according to rule of 5(6)/0
    /// is_trivial<T> has to be true
    static_assert(std::is_default_constructible_v<T>, "element is not default-constructible!");
    static_assert(std::is_trivially_copyable_v<T>, "element is not trivially-copyable!");
    using base_t = vector_instance<T, Index>;
    using const_base_t = const vector_instance<T, Index>;
    using value_type = remove_cvref_t<T>;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = Index;
    using difference_type = std::make_signed_t<size_type>;
    using iterator_category = std::random_access_iterator_tag;  // std::contiguous_iterator_tag;

    constexpr MemoryHandle &base() noexcept { return static_cast<MemoryHandle &>(*this); }
    constexpr const MemoryHandle &base() const noexcept {
      return static_cast<const MemoryHandle &>(*this);
    }
    constexpr base_t &self() noexcept { return static_cast<base_t &>(*this); }
    constexpr const base_t &self() const noexcept { return static_cast<const base_t &>(*this); }

    constexpr Vector(memsrc_e mre = memsrc_e::host, ProcID devid = -1,
                     std::size_t alignment = std::alignment_of_v<T>)
        : base_t{buildInstance(mre, devid, 0)},
          MemoryHandle{mre, devid},
          _size{0},
          _align{alignment} {}
    explicit Vector(size_type count, memsrc_e mre = memsrc_e::host, ProcID devid = -1,
                    std::size_t alignment = std::alignment_of_v<T>)
        : base_t{buildInstance(mre, devid, count)},
          MemoryHandle{mre, devid},
          _size{count},
          _align{alignment} {}
#if 0
    explicit Vector(std::initializer_list<T> vals)
        : Vector{vals.size(), memsrc_e::host, -1, std::alignment_of_v<T>} {
      size_type i = 0;
      for (const auto &src : vals) (*this)[i++] = src;
    }
    Vector(const std::vector<T> &vals)
        : Vector{vals.size(), memsrc_e::host, -1, std::alignment_of_v<T>} {
      size_type i = 0;
      for (const auto &src : vals) (*this)[i++] = src;
    }
#endif

    ~Vector() {
      if (data() && capacity() > 0) self().dealloc();
    }

    struct iterator : IteratorInterface<iterator> {
      template <typename Ti> constexpr iterator(const base_t &range, Ti &&idx)
          : _range{range}, _idx{static_cast<size_type>(idx)} {}

      constexpr reference dereference() { return _range(_idx); }
      constexpr bool equal_to(iterator it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(iterator it) const noexcept { return it._idx - _idx; }

    protected:
      base_t _range{};
      size_type _idx{0};
    };
    struct const_iterator : IteratorInterface<const_iterator> {
      template <typename Ti> constexpr const_iterator(const base_t &range, Ti &&idx)
          : _range{range}, _idx{static_cast<size_type>(idx)} {}

      constexpr const_reference dereference() { return _range(_idx); }
      constexpr bool equal_to(const_iterator it) const noexcept { return it._idx == _idx; }
      constexpr void advance(difference_type offset) noexcept { _idx += offset; }
      constexpr difference_type distance_to(const_iterator it) const noexcept {
        return it._idx - _idx;
      }

    protected:
      base_t _range{};
      size_type _idx{0};
    };

    constexpr auto begin() noexcept { return make_iterator<iterator>(self(), 0); }
    constexpr auto end() noexcept { return make_iterator<iterator>(self(), size()); }
    constexpr auto begin() const noexcept { return make_iterator<const_iterator>(self(), 0); }
    constexpr auto end() const noexcept { return make_iterator<const_iterator>(self(), size()); }

    void debug() const {
      fmt::print("procid: {}, memspace: {}, size: {}, capacity: {}\n", static_cast<int>(devid()),
                 static_cast<int>(memspace()), size(), capacity());
    }

    /// capacity
    constexpr size_type size() const noexcept { return _size; }
    constexpr size_type capacity() const noexcept { return self().node().extent(); }
    constexpr bool empty() noexcept { return size() == 0; }
    constexpr pointer head() const noexcept { return reinterpret_cast<pointer>(self().address()); }
    constexpr pointer tail() const noexcept { return reinterpret_cast<pointer>(head() + size()); }

    /// element access
    constexpr reference operator[](size_type idx) noexcept { return self()(idx); }
    constexpr conditional_t<std::is_fundamental_v<value_type>, value_type, const_reference>
    operator[](size_type idx) const noexcept {
      return self()(idx);
    }
    /// ctor, assignment operator
    Vector(const Vector &o)
        : base_t{buildInstance(o.memspace(), o.devid(), o.capacity())},
          MemoryHandle{o.base()},
          _size{o.size()},
          _align{o._align} {
      if (o.data())
        copy(MemoryEntity{base(), (void *)data()}, MemoryEntity{o.base(), (void *)o.data()},
             o.usedBytes());
    }
    Vector &operator=(const Vector &o) {
      if (this == &o) return *this;
      Vector tmp(o);
      swap(tmp);
      return *this;
    }
    Vector clone(const MemoryHandle &mh) const {
      Vector ret{capacity(), mh.memspace(), mh.devid(), _align};
      copy({mh, (void *)ret.data()}, {base(), (void *)this->data()}, usedBytes());
      return ret;
    }
    /// assignment or destruction after std::move
    /// https://www.youtube.com/watch?v=ZG59Bqo7qX4
    /// explicit noexcept
    /// leave the source object in a valid (default constructed) state
    Vector(Vector &&o) noexcept {
      const Vector defaultVector{};
      base() = std::exchange(o.base(), defaultVector.base());
      self() = std::exchange(o.self(), defaultVector.self());
      _size = std::exchange(o._size, defaultVector.size());
      _align = std::exchange(o._align, defaultVector._align);
    }
    /// make move-assignment safe for self-assignment
    Vector &operator=(Vector &&o) noexcept {
      if (this == &o) return *this;
      Vector tmp(std::move(o));
      swap(tmp);
      return *this;
    }
    void swap(Vector &o) noexcept {
      base().swap(o.base());
      // fmt::print("before swap: self {}, o {}\n", self().address(), o.self().address());
      std::swap(self(), o.self());
      // fmt::print("after swap: self {}, o {}\n", self().address(), o.self().address());
      std::swap(_size, o._size);
      std::swap(_align, o._align);
    }

    // constexpr operator base_t &() noexcept { return self(); }
    // constexpr operator base_t() const noexcept { return self(); }
    // void relocate(memsrc_e mre, ProcID devid) {}
    void clear() { resize(0); }
    void resize(size_type newSize) {
      const auto oldSize = size();
      if (newSize < oldSize) {
        if constexpr (!std::is_trivially_destructible_v<T>) {
          static_assert(!std::is_trivial_v<T>, "should not activate this scope");
          pointer ed = tail();
          for (pointer e = head() + newSize; e < ed; ++e) e->~T();
        }
        _size = newSize;
        return;
      }
      if (newSize > oldSize) {
        const auto oldCapacity = capacity();
        if (newSize > oldCapacity) {
          auto &rm = get_resource_manager();
          base_t tmp{buildInstance(memspace(), devid(), geometric_size_growth(newSize))};
          if (size()) copy({base(), (void *)tmp.address()}, {base(), (void *)data()}, usedBytes());
          if (oldCapacity > 0 && data()) self().dealloc();  // rm.deallocate((void *)data());
          self() = tmp;
          _size = newSize;
          return;
        }
      }
    }

    void push_back(const value_type &val) {
      if (size() >= capacity()) resize(size() + 1);
      (*this)(_size++) = val;
    }
    void push_back(value_type &&val) {
      if (size() >= capacity()) resize(size() + 1);
      (*this)(_size++) = std::move(val);
    }

    template <typename InputIter> iterator append(InputIter st, InputIter ed) {
      // difference_type count = std::distance(st, ed); //< def standard iterator
      difference_type count = ed - st;
      if (count <= 0) return end();
      size_type unusedCapacity = capacity() - size();
      // this is not optimal
      if (count > unusedCapacity) {
        resize(size() + count);
      } else {
        _size += count;
      }
      copy({base(), (void *)&(*end())}, {base(), (void *)&(*st)}, sizeof(T) * count);
      return end();
    }
    iterator append(const Vector &other) { return append(other.begin(), other.end()); }
    constexpr const_pointer data() const noexcept { return (const_pointer)head(); }
    constexpr pointer data() noexcept { return (pointer)head(); }
    constexpr reference front() noexcept { return (*this)(0); }
    constexpr const_reference front() const noexcept { (*this)(0); }
    constexpr reference back() noexcept { return (*this)(size() - 1); }
    constexpr const_reference back() const noexcept { (*this)(size() - 1); }

  protected:
    constexpr std::size_t usedBytes() const noexcept { return sizeof(T) * size(); }

    constexpr auto buildInstance(memsrc_e mre, ProcID devid, size_type capacity) const {
      using namespace ds;
      constexpr auto dec = ds::static_decorator<>{};
      uniform_domain<0, size_type, 1, index_seq<0>> dom{wrapv<0>{}, capacity};
      vector_snode<wrapt<T>, size_type> node{dec, dom, zs::make_tuple(wrapt<T>{}), vseq_t<1>{}};
      auto inst = instance{wrapv<dense>{}, zs::make_tuple(node)};
      if (capacity) inst.alloc(get_memory_source(mre, devid));
      return inst;
    }
    constexpr size_type geometric_size_growth(size_type newSize) noexcept {
      size_type geometricSize = capacity();
      geometricSize = geometricSize + geometricSize / 2;
      if (newSize > geometricSize) return newSize;
      return geometricSize;
    }

    size_type _size{0};  // size
    size_type _align{0};
  };

  template <execspace_e, typename VectorT, typename = void> struct VectorProxy {
    using vector_t = typename VectorT::pointer;
    using size_type = typename VectorT::size_type;

    constexpr VectorProxy() = default;
    ~VectorProxy() = default;
    explicit constexpr VectorProxy(VectorT &vector)
        : _vector{vector.data()}, _vectorSize{vector.size()} {}

    constexpr decltype(auto) operator[](size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator[](size_type i) const { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }

    size_type _vectorSize;
    vector_t _vector;
  };

  template <execspace_e Space, typename VectorT> struct VectorProxy<Space, const VectorT> {
    using vector_t = typename VectorT::const_pointer;
    using size_type = typename VectorT::size_type;

    constexpr VectorProxy() = default;
    ~VectorProxy() = default;
    explicit constexpr VectorProxy(const VectorT &vector)
        : _vector{vector.data()}, _vectorSize{vector.size()} {}

    constexpr decltype(auto) operator[](size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator[](size_type i) const { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) { return _vector[i]; }
    constexpr decltype(auto) operator()(size_type i) const { return _vector[i]; }

    size_type _vectorSize;
    vector_t _vector;
  };

  template <execspace_e ExecSpace, typename T, typename Index>
  constexpr decltype(auto) proxy(Vector<T, Index> &vec) {  // currently ignore constness
    return VectorProxy<ExecSpace, Vector<T, Index>>{vec};
  }
  template <execspace_e ExecSpace, typename T, typename Index>
  constexpr decltype(auto) proxy(const Vector<T, Index> &vec) {  // currently ignore constness
    return VectorProxy<ExecSpace, const Vector<T, Index>>{vec};
  }

}  // namespace zs