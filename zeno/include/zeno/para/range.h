#pragma once

#include <iterator>
#include <tuple>

namespace zeno {

template <class T, class = void>
struct range_traits : std::false_type {};

template <class T>
struct range_traits<T, std::void_t<decltype(
    std::begin(std::declval<T>()) != std::end(std::declval<T>())
    )>> : std::true_type {
    using iterator_type = decltype(std::begin(std::declval<T>()));
    using sentinel_type = decltype(std::end(std::declval<T>()));
};

template <class T>
constexpr auto is_range_v = range_traits<T>::value;

template <class T>
using range_iterator_t = typename range_traits<T>::iterator_type;

template <class It>
struct range {
    It b;
    It e;

    range(It b_, It e_)
        : b(std::move(b_)), e(std::move(e_))
    {}

    explicit range(std::pair<It, It> const &be_)
        : range(be_.first, be_.second) {}

    template <class T, class = std::enable_if_t<is_range_v<T>>>
    range(T &&t_)
        : range(std::begin(t_), std::end(t_)) {}

    It begin() const {
        return b;
    }

    It end() const {
        return e;
    }

    range &operator++() const {
        b = std::next(b);
        return *this;
    }

    range &operator+=(std::ptrdiff_t n) const {
        std::advance(b, n);
        return *this;
    }

    decltype(auto) operator*() const {
        return *b;
    }

    void removePrefix(std::ptrdiff_t n) const {
        b += n;
    }

    void removeSuffix(std::ptrdiff_t n) const {
        e -= n;
    }
};


template <class It>
range(It, It) -> range<It>;

template <class It>
range(std::pair<It, It>) -> range<It>;

template <class T, class = std::enable_if_t<is_range_v<T>>>
range(T &&) -> range<range_iterator_t<T>>;

}
