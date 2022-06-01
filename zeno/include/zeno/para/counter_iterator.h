#pragma once

#include <iterator>
#include <tuple>
#include <zeno/para/iterator_facade.h>

namespace zeno {

template <class T>
struct counter_iterator : iterator_facade<counter_iterator<T>
, T
, std::random_access_iterator_tag
, T const &
, T const *
, std::ptrdiff_t
> {
    T counter;

    explicit counter_iterator(T counter) : counter(std::move(counter)) {}

    T dereference() const {
        return counter;
    }

    void increment() {
        ++counter;
    }

    void decrement() {
        --counter;
    }

    void advance(std::ptrdiff_t n) {
        counter += n;
    }

    std::ptrdiff_t distance_to(counter_iterator const &that) const {
        return that.counter - counter;
    }

    bool equal_to(counter_iterator const &that) const {
        return counter == that.counter;
    }
};

template <class T>
counter_iterator(T) -> counter_iterator<T>;

static_assert(std::is_same_v<typename std::iterator_traits<counter_iterator<int>>::iterator_category, std::random_access_iterator_tag>);

}
