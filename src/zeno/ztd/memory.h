#pragma once

#include <zeno/common.h>
#include <memory>

ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _memory_h {


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


using generic_ptr = std::shared_ptr<void>;

template <class T>
inline generic_ptr make_generic(auto &&...args) {
    return std::make_shared<T>(std::forward<decltype(args)>(args)...);
}

template <class T>
inline std::shared_ptr<T> cast(generic_ptr p) {
    return std::dynamic_pointer_cast<T>(p);
}


}
}
ZENO_NAMESPACE_END
