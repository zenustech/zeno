#pragma once

#include <memory>

namespace zeno::ztd {


template <class T>
struct stale_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;

    ~stale_ptr() {
        std::unique_ptr<T>::release();
    }
};

template <class T>
stale_ptr(T *) -> stale_ptr<T>;





template <class T>
struct copiable_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;
    using std::unique_ptr<T>::operator=;

    copiable_ptr(std::unique_ptr<T> &&o)
        : std::unique_ptr<T>(std::move(o)) {
    }

    copiable_ptr(copiable_ptr const &o)
        : std::unique_ptr<T>(std::make_unique<T>(
            static_cast<T const &>(*o))) {
    }

    copiable_ptr &operator=(copiable_ptr const &o) {
        std::unique_ptr<T>::operator=(std::unique_ptr<T>(
            std::make_unique<T>(static_cast<T const &>(*o))));
        return *this;
    }

    operator std::unique_ptr<T> &() { return *this; }
    operator std::unique_ptr<T> const &() const { return *this; }
};


template <class T>
copiable_ptr(T *) -> copiable_ptr<T>;

template <class T>
copiable_ptr(std::unique_ptr<T> &&) -> copiable_ptr<T>;



}
