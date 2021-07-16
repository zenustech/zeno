#include <type_traits>
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

template <class T>
struct iterator_base {
    //static_assert(std::is_base_of_v<iterator_base, T>);

    inline T const &begin() const {
        return *reinterpret_cast<T const *>(this);
    }

    inline auto end() const {
        return npos;
    }

    template <class _ = void>
    void next(int skip) {
        static_assert(sizeof(_), "next(int) not implemented");
    }

    template <class _ = void>
    void prev(int skip) {
        static_assert(sizeof(_), "prev(int) not implemented");
    }

    template <class _ = void>
    void eof() const {
        static_assert(sizeof(_), "eof() not implemented");
    }

    template <class _ = void>
    void get() const {
        static_assert(sizeof(_), "get() not implemented");
    }

    inline auto &get() {
        return const_cast<T const *>(reinterpret_cast<T *>(this))->get();
    }

    inline T &operator++() {
        that()->next(1);
        return *that();
    }

    inline T &operator--() {
        that()->prev(1);
        return *that();
    }

    inline T operator++(int) {
        auto old = *that();
        that()->next(1);
        return old;
    }

    inline T operator--(int) {
        auto old = *that();
        that()->prev(1);
        return old;
    }

    inline T &operator+=(int skip) {
        that()->next(skip);
        return *that();
    }

    inline T &operator-=(int skip) {
        that()->prev(skip);
        return *that();
    }

    inline operator bool() const {
        return that()->alive();
    }

    inline decltype(auto) operator*() const {
        return that()->get();
    }

    inline decltype(auto) operator*() {
        return that()->get();
    }

private:
    T *that() {
        return reinterpret_cast<T *>(this);
    }

    T const *that() const {
        return reinterpret_cast<T const *>(this);
    }
};

template <class T>
struct range : iterator_base<range<T>> {
    using value_type = T;

    T m_now;
    T m_end;

    range
        ( T const &now_
        , T const &end_
        )
    : m_now(now_)
    , m_end(end_)
    {}

    void next(int skip) {
        m_now += skip;
    }

    void prev(int skip) {
        m_now -= skip;
    }

    bool alive() const {
        return m_now < m_end;
    }

    value_type const &get() const {
        return m_now;
    }
};

template <class T>
struct slice : iterator_base<slice<T>> {
    using value_type = typename T::value_type;

    T m_iter;
    value_type m_begin;
    value_type m_end;

    slice
        ( T const &iter
        , value_type const &begin_
        , value_type const &end_
        )
        : m_iter(iter)
        , m_begin(begin_)
        , m_end(end_)
        {}

    void next(int skip) {
        m_iter += skip;
    }

    bool alive() const {
        return m_iter;
    }

    auto get() const {
        return *m_iter;
    }
};


int main(void)
{
    auto r = range(2, 15);
    auto s = slice(r, 2, 4);
    for (auto i: s) show(i);
}
