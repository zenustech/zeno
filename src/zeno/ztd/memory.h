#pragma once

#include <memory>

namespace zeno::ztd {

template <class T>
struct stale_unique_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;

    ~stale_unique_ptr() {
        std::unique_ptr<T>::release();
    }
};

template <class T>
stale_unique_ptr(T *) -> stale_unique_ptr<T>;

}
