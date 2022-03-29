#pragma once

#include <iterator>
#include <tuple>

namespace zeno {

template <class T0, class ...Ts>
struct zip_iterator {
    using iterator_category = typename std::iterator_traits<T0>::iterator_category;
    using difference_type = typename std::iterator_traits<T0>::difference_type;
    using value_type = std::tuple<decltype(*std::declval<T0 const &>()), decltype(*std::declval<Ts const &>())...>;

    T0 t0;
    std::tuple<Ts...> ts;

    explicit zip_iterator(T0 const &t0, Ts const &...ts) : t0(t0), ts(ts...) {}

    template <std::size_t ...Is>
    value_type _dereference_t0ts(std::index_sequence<Is...>) {
        return value_type{*t0, *std::get<Is>(ts)...};
    }

    constexpr value_type operator*() const {
        return _dereference_t0ts(std::make_index_sequence<sizeof...(Ts)>{});
    }

    template <std::size_t ...Is>
    void _plusplus_ts(std::index_sequence<Is...>) {
        ((void)++std::get<Is>(ts), ...);
    }

    template <class...>
    zip_iterator &operator++() {
        ++t0;
        _plusplus_ts(std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

    zip_iterator operator++(int) {
        auto that = *this;
        this->operator++();
        return that;
    }

    template <std::size_t ...Is>
    void _minusminus_ts(std::index_sequence<Is...>) {
        ((void)--std::get<Is>(ts), ...);
    }

    template <class...>
    zip_iterator &operator--() {
        --t0;
        _minusminus_ts(std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

    zip_iterator operator--(int) {
        auto that = *this;
        this->operator--();
        return that;
    }

    template <std::size_t ...Is>
    void _plusn_ts(difference_type n, std::index_sequence<Is...>) {
        (void(std::get<Is>(ts) += n), ...);
    }

    template <class...>
    zip_iterator &operator+=(difference_type n) {
        t0 += n;
        _plusn_ts(n, std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

    template <std::size_t ...Is>
    void _minusn_ts(difference_type n, std::index_sequence<Is...>) {
        (void(std::get<Is>(ts) -= n), ...);
    }

    template <class...>
    zip_iterator &operator-=(difference_type n) {
        t0 -= n;
        _minusn_ts(n, std::make_index_sequence<sizeof...(Ts)>{});
        return *this;
    }

    zip_iterator operator+(difference_type n) const {
        auto that = *this;
        that.operator+=(n);
        return that;
    }

    zip_iterator operator-(difference_type n) const {
        auto that = *this;
        that.operator-=(n);
        return that;
    }

    template <class...>
    zip_iterator operator-(zip_iterator const &that) const {
        return t0 - that.t0;
    }

    template <class...>
    bool operator==(zip_iterator const &that) const {
        return t0 == that.t0;
    }

    template <class...>
    bool operator!=(zip_iterator const &that) const {
        return !this->operator==(that);
    }
};

template <class T, class ...Ts>
zip_iterator(T const &, Ts const &...) -> zip_iterator<T, Ts...>;

}
