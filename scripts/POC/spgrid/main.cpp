#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <array>
#include <cassert>
#include <omp.h>

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)


struct null_iterator {
    template <class T>
    bool operator!=(T const &other) const {
        return (bool)other;
    }

    template <class T>
    bool operator==(T const &other) const {
        return !(bool)other;
    }
};

template <class T>
inline bool operator!=(T const &other, null_iterator) {
    return (bool)other;
}

template <class T>
inline bool operator==(T const &other, null_iterator) {
    return !(bool)other;
}

inline constexpr null_iterator npos{};

template <class D, class T>
struct iterator_base {
    inline D const &begin() const {
        return *static_cast<D const *>(this);
    }

    inline auto end() const {
        return npos;
    }

    template <class _ = void>
    [[noreturn]] void next() {
        static_assert(sizeof(_), "next() not implemented");
    }

    template <class _ = void>
    [[noreturn]] void prev() {
        static_assert(sizeof(_), "prev() not implemented");
    }

    template <class _ = void>
    [[noreturn]] bool eof() const {
        static_assert(sizeof(_), "eof() not implemented");
    }

    template <class _ = void>
    [[noreturn]] T const &get() const {
        static_assert(sizeof(_), "get() not implemented");
    }

    template <class = void>
    T &get() {
        return const_cast<D const *>(static_cast<D *>(this))->get();
    }

    template <class = void>
    D &operator++() {
        static_cast<D *>(this)->next();
        return *static_cast<D *>(this);
    }

    template <class = void>
    D &operator--() {
        static_cast<D *>(this)->prev();
        return *static_cast<D *>(this);
    }

    template <class = void>
    operator bool() const {
        return static_cast<D const *>(this)->eof();
    }

    template <class = void>
    decltype(auto) operator*() const {
        return static_cast<D const *>(this)->get();
    }

    template <class = void>
    decltype(auto) operator*() {
        return static_cast<D *>(this)->get();
    }
};

template <class T>
struct range : iterator_base<range<T>, T> {
    T m_now;
    T m_end;

    range
        ( T const &now_
        , T const &end_
        )
    : m_now(now_)
    , m_end(end_)
    {}

    void next() {
        m_now++;
    }

    bool eof() const {
        return m_now != m_end;
    }

    T const &get() const {
        return m_now;
    }
};

template <class T>
struct slice {
    T t;

    slice
        ( T const &t
        )
        : t(t)
        {}
};


int main(void)
{
    auto r = range(2, 4);
    //for (auto it = r.begin(); it; ++it) show(*it);
    for (auto i: r) show(i);
}
