#pragma once

#include <iterator>
#include <tuple>

namespace zeno {

template <class T>
struct counter_iterator {
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using pointer = T const *;
    using reference = T const &;
    using difference_type = std::ptrdiff_t;

    T counter;

    explicit counter_iterator(T counter) : counter(std::move(counter)) {}

    constexpr value_type operator*() const {
        return counter;
    }

    constexpr pointer operator->() const {
        return &counter;
    }

    counter_iterator &operator++() {
        ++counter;
        return *this;
    }

    counter_iterator operator++(int) {
        auto that = *this;
        this->operator++();
        return that;
    }

    counter_iterator &operator+=(difference_type n) {
        counter += n;
        return *this;
    }

    counter_iterator &operator-=(difference_type n) {
        counter -= n;
        return *this;
    }

    counter_iterator operator+(difference_type n) const {
        auto that = *this;
        that.operator+=(n);
        return that;
    }

    counter_iterator operator-(difference_type n) const {
        auto that = *this;
        that.operator-=(n);
        return that;
    }

    counter_iterator operator-(counter_iterator that) const {
        return counter - that.counter;
    }
};

template <class T>
counter_iterator(T) -> counter_iterator<T>;

}
