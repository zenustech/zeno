#pragma once

#include <iterator>
#include <tuple>
#include <zeno/para/iterator_facade.h>

namespace zeno {

template <class T0, class ...Ts>
struct zip_iterator : iterator_facade<zip_iterator<T0, Ts...>
, std::tuple<decltype(*std::declval<T0 const &>()), decltype(*std::declval<Ts const &>())...>
, typename std::iterator_traits<T0>::iterator_category
, typename std::iterator_traits<T0>::reference
, typename std::iterator_traits<T0>::difference_type
> {
public:
    using value_type = std::tuple<decltype(*std::declval<T0 const &>()), decltype(*std::declval<Ts const &>())...>;
    using iterator_category = typename std::iterator_traits<T0>::iterator_category;
    using reference = typename std::iterator_traits<T0>::reference;
    using difference_type = typename std::iterator_traits<T0>::difference_type;

private:
    T0 t0;
    std::tuple<Ts...> ts;

public:
    explicit zip_iterator(T0 const &t0, Ts const &...ts) : t0(t0), ts(ts...) {}

private:
    template <std::size_t ...Is>
    value_type _dereference_t0ts(std::index_sequence<Is...>) {
        return value_type{*t0, *std::get<Is>(ts)...};
    }

public:
    template <class...>
    value_type dereference() const {
        return _dereference_t0ts(std::make_index_sequence<sizeof...(Ts)>{});
    }

private:
    template <std::size_t ...Is>
    void _increment_ts(std::index_sequence<Is...>) {
        ((void)++std::get<Is>(ts), ...);
    }

public:
    template <class...>
    void increment() {
        ++t0;
        _increment_ts(std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

private:
    template <std::size_t ...Is>
    void _decrement_ts(std::index_sequence<Is...>) {
        ((void)--std::get<Is>(ts), ...);
    }

public:
    template <class...>
    void decrement() {
        --t0;
        _decrement_ts(std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

private:
    template <std::size_t ...Is>
    void _advance_ts(difference_type n, std::index_sequence<Is...>) {
        (void(std::get<Is>(ts) += n), ...);
    }

public:
    template <class...>
    void advance(difference_type n) {
        t0 += n;
        _advance_ts(n, std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

    template <class...>
    zip_iterator distance_to(difference_type n) const {
        auto that = *this;
        that.operator-=(n);
        return that;
    }

    template <class...>
    difference_type distance_to(zip_iterator const &that) const {
        return t0 - that.t0;
    }

    template <class...>
    bool equal_to(zip_iterator const &that) const {
        return t0 == that.t0;
    }
};

template <class T, class ...Ts>
zip_iterator(T const &, Ts const &...) -> zip_iterator<T, Ts...>;

}
