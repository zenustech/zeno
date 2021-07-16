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

template <class D>
struct iterator_base {
    inline D const &begin() const {
        return *reinterpret_cast<D const *>(this);
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
    [[noreturn]] auto const &get() const {
        static_assert(sizeof(_), "get() not implemented");
    }

    inline auto &get() {
        return const_cast<D const *>(reinterpret_cast<D *>(this))->get();
    }

    inline D &operator++() {
        reinterpret_cast<D *>(this)->next();
        return *reinterpret_cast<D *>(this);
    }

    inline D &operator--() {
        reinterpret_cast<D *>(this)->prev();
        return *reinterpret_cast<D *>(this);
    }

    inline operator bool() const {
        return reinterpret_cast<D const *>(this)->eof();
    }

    inline decltype(auto) operator*() const {
        return reinterpret_cast<D const *>(this)->get();
    }

    inline decltype(auto) operator*() {
        return reinterpret_cast<D *>(this)->get();
    }
};

template <class T>
struct range : iterator_base<range<T>> {
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

    void next() {
        t++;
    }

    bool eof() const {
        return t;
    }

    T const &get() const {
        return *t;
    }
};


int main(void)
{
    auto r = range(2, 4);
    //for (auto it = r.begin(); it; ++it) show(*it);
    for (auto i: r) show(i);
}
