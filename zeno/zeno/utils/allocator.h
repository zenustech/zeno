#pragma once

#include <new>
#include <utility>
#include <cstddef>
#include <type_traits>
#include <memory>

namespace zeno {
inline namespace allocator_h {

template <class T, std::align_val_t Align = 64, bool NoInit = false>
struct allocator {
    using size_type = std::size_t;
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;

    allocator() = default;

    T *allocate(size_type n, void *p = 0) {
        return reinterpret_cast<T *>(::new (Align) std::byte[sizeof(T)]);
    }

    T *deallocate(T *p, size_type n) {
        ::operator delete(reinterpret_cast<void *>(p), Align);
    }

    template <class U, class ...Args>
    constexpr void construct(U *p, Args &&...args) const {
        if constexpr (!(NoInit && sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }

    template <class U, std::align_val_t UAlign>
    constexpr bool operator==(allocator<U, UAlign> const &) noexcept {
        return Align == UAlign;
    }

    template <class U, std::align_val_t UAlign>
    constexpr bool operator!=(allocator<U, UAlign> const &) noexcept {
        return Align != UAlign;
    }
};

}
}
