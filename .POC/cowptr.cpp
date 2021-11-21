#include <iostream>
#include <memory>


template <class T>
struct cow {
    std::shared_ptr<T> _M_p;

    cow() = default;
    cow(cow const &) = default;
    cow &operator=(cow const &) = default;
    cow(cow &&) = default;
    cow &operator=(cow &&) = default;

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


int main() {
    auto x = int(42);
    std::cout << *x << std::endl;
    return 0;
}
