#pragma once

#include <memory>
#include <zeno/common.h>


ZENO_NAMESPACE_BEGIN
namespace ztd {

template <class T>
struct cow {
    std::shared_ptr<T> _M_p;

    cow() = default;
    cow(cow const &) = default;
    cow &operator=(cow const &) = default;
    cow(cow &&) = default;
    cow &operator=(cow &&) = default;

    cow(std::shared_ptr<T> &&p)
        : _M_p(std::move(p)) {}
    cow(std::shared_ptr<T> const &p)
        : _M_p(p) {}

    inline T const *rget() const {
        return _M_p.get();
    }

    T *wget() {
        if (_M_p)
            _M_p = std::make_shared<T>(static_cast<T const &>(*_M_p));
        return _M_p.get();
    }

    inline operator T const *() const {
        return rget();
    }

    inline explicit operator T *() {
        return wget();
    }
};

template <class T>
inline cow<T> make_cow(auto &&...args) {
    return std::make_shared<T>(std::forward<decltype(args)>(args)...);
}

}
ZENO_NAMESPACE_END
