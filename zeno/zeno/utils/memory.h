#pragma once


#include <memory>

namespace zeno {


template <class T>
struct copiable_unique_ptr : std::unique_ptr<T> {
    using std::unique_ptr<T>::unique_ptr;
    using std::unique_ptr<T>::operator=;

    copiable_unique_ptr() = default;
    ~copiable_unique_ptr() = default;

    copiable_unique_ptr &operator=(copiable_unique_ptr const &o) {
        std::unique_ptr<T>::operator=(std::unique_ptr<T>(
            o ? std::make_unique<T>(static_cast<T const &>(*o)) : nullptr));
        return *this;
    }

    copiable_unique_ptr &operator=(copiable_unique_ptr &&o) {
        std::unique_ptr<T>::operator=(std::move(o));
        return *this;
    }

    copiable_unique_ptr &operator=(std::unique_ptr<T> &&o) {
        std::unique_ptr<T>::operator=(std::move(o));
        return *this;
    }

    copiable_unique_ptr(std::unique_ptr<T> &&o)
        : std::unique_ptr<T>(std::move(o)) {
    }

    copiable_unique_ptr(copiable_unique_ptr &&o)
        : std::unique_ptr<T>(std::move(o)) {
    }

    copiable_unique_ptr(copiable_unique_ptr const &o)
        : std::unique_ptr<T>(o ? std::make_unique<T>(
            static_cast<T const &>(*o)) : nullptr) {
    }

    template <bool lazyHelper = true>
    T &access() const {
        if constexpr (lazyHelper) {
            if (!std::unique_ptr<T>::operator bool())
                const_cast<copiable_unique_ptr *>(this)->
                    std::unique_ptr<T>::operator=(std::make_unique<T>());
        } else {
            static_assert(lazyHelper);
        }
        return *std::unique_ptr<T>::get();
    }

    operator std::unique_ptr<T> &() { return *this; }
    operator std::unique_ptr<T> const &() const { return *this; }
};


template <class T>
copiable_unique_ptr(std::unique_ptr<T> &&o) -> copiable_unique_ptr<T>;


}
