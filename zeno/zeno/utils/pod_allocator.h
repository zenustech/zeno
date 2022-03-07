#pragma once

#include <new>
#include <utility>

namespace zeno {

template <class AllocT>
struct pod_allocator : AllocT {
    template <class T, class ...Args>
    void construct(T *p, Args &&...args) const {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }
};

}
