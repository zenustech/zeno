#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <memory>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace zeno {
inline namespace aligned_allocator_h {


// https://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory
namespace detail {
    void* allocate_aligned_memory(size_t align, size_t size) {
#if defined(_WIN32)
        return _aligned_malloc(size, align);
#else
        return std::aligned_alloc(align, size);
#endif
    }
    void deallocate_aligned_memory(void* ptr) noexcept {
#if defined(_WIN32)
        _aligned_free(size, align);
#else
        std::free(ptr);
#endif
    }
}

template <typename T, size_t Align = 64>
class aligned_allocator;

template <size_t Align>
class aligned_allocator<void, Align>
{
public:
    typedef void*             pointer;
    typedef const void*       const_pointer;
    typedef void              value_type;

    template <class U> struct rebind { typedef aligned_allocator<U, Align> other; };
};


template <typename T, size_t Align>
class aligned_allocator
{
public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef std::true_type propagate_on_container_move_assignment;

    template <class U>
    struct rebind { typedef aligned_allocator<U, Align> other; };

public:
    aligned_allocator() noexcept
    {}

    template <class U>
    aligned_allocator(const aligned_allocator<U, Align>&) noexcept
    {}

    size_type
    max_size() const noexcept
    { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    pointer
    address(reference x) const noexcept
    { return std::addressof(x); }

    const_pointer
    address(const_reference x) const noexcept
    { return std::addressof(x); }

    pointer
    allocate(size_type n, typename aligned_allocator<void, Align>::const_pointer = 0)
    {
        const size_type alignment = static_cast<size_type>( Align );
        void* ptr = detail::allocate_aligned_memory(alignment , n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void
    deallocate(pointer p, size_type) noexcept
    { return detail::deallocate_aligned_memory(p); }

    template <class U, class ...Args>
    void
    construct(U* p, Args&&... args)
    { ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...); }

    void
    destroy(pointer p)
    { p->~T(); }
};


template <typename T, size_t Align>
class aligned_allocator<const T, Align>
{
public:
    typedef T         value_type;
    typedef const T*  pointer;
    typedef const T*  const_pointer;
    typedef const T&  reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef std::true_type propagate_on_container_move_assignment;

    template <class U>
    struct rebind { typedef aligned_allocator<U, Align> other; };

public:
    aligned_allocator() noexcept
    {}

    template <class U>
    aligned_allocator(const aligned_allocator<U, Align>&) noexcept
    {}

    size_type
    max_size() const noexcept
    { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    const_pointer
    address(const_reference x) const noexcept
    { return std::addressof(x); }

    pointer
    allocate(size_type n, typename aligned_allocator<void, Align>::const_pointer = 0)
    {
        const size_type alignment = static_cast<size_type>( Align );
        void* ptr = detail::allocate_aligned_memory(alignment , n * sizeof(T));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<pointer>(ptr);
    }

    void
    deallocate(pointer p, size_type) noexcept
    { return detail::deallocate_aligned_memory(p); }

    template <class U, class ...Args>
    void
    construct(U* p, Args&&... args)
    { ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...); }

    void
    destroy(pointer p)
    { p->~T(); }
};

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline
bool
operator== (const aligned_allocator<T,TAlign>&, const aligned_allocator<U, UAlign>&) noexcept
{ return TAlign == UAlign; }

template <typename T, size_t TAlign, typename U, size_t UAlign>
inline
bool
operator!= (const aligned_allocator<T,TAlign>&, const aligned_allocator<U, UAlign>&) noexcept
{ return TAlign != UAlign; }

}
}
