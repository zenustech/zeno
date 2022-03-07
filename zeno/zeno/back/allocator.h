#pragma once

#include <new>
#include <utility>
#include <cstddef>
#include <type_traits>
#include <memory>

namespace zeno {

template <class T, std::size_t Align = 64, bool Pod = true>
struct allocator {
    using size_type = std::size_t;
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;

    template <class U>
    struct rebind {
        using other = allocator<U, Align, Pod>;
    };

    allocator() = default;

    static T *allocate(size_type n) {
        n *= sizeof(T);
        return reinterpret_cast<T *>(::operator new(n, std::align_val_t(Align)));
    }

    static void deallocate(T *p, size_type = 0) {
        ::operator delete(reinterpret_cast<void *>(p), std::align_val_t(Align));
    }

    template <class U, class ...Args>
    constexpr static void construct(U *p, Args &&...args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>) {
        if constexpr (!(Pod && std::is_pod_v<T> && sizeof...(Args) == 0))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }

    template <class U, std::size_t UAlign, bool UPod>
    constexpr bool operator==(allocator<U, UAlign, Pod> const &) noexcept {
        return Align == UAlign && Pod == UPod;
    }

    template <class U, std::size_t UAlign, bool UPod>
    constexpr bool operator!=(allocator<U, UAlign, Pod> const &) noexcept {
        return Align != UAlign || Pod != UPod;
    }
};

}
