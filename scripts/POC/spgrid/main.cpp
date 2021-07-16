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
struct range {
    T m_begin;
    T m_end;

    range
        ( T const &begin_
        , T const &end_
        )
    : m_begin(begin_)
    , m_end(end_)
    {}

    struct iterator : std::iterator<std::forward_iterator_tag, T> {
        T m_now;
        T m_end;

        iterator
            ( T const &now_
            , T const &end_
            )
        : m_now(now_)
        , m_end(end_)
        {}

        iterator &operator++() {
            m_now++;
            return *this;
        }

        operator bool() const {
            return m_now != m_end;
        }

        T const &operator*() const {
            return m_now;
        }
    };

    iterator begin() const {
        return iterator{m_begin, m_end};
    }

    auto end() const {
        return npos;
    }
};


int main(void)
{
    auto r = range(2, 4);
    //for (auto it = r.begin(); it; ++it) show(*it);
    for (auto i: r) show(i);
}
