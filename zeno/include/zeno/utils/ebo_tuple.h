#pragma once

#include <variant>
#include <type_traits>
#include <utility>

namespace zeno {

template <class Tup, class Seq>
struct _impl_ebo_tuple {
    _impl_ebo_tuple() = delete;
};

template <std::size_t I, class T>
struct _impl_ebo_entry : protected T {
    _impl_ebo_entry() = default;

    _impl_ebo_entry(T &&t) : T(std::move(t)) {
    }
};

template <class Tup, std::size_t ...Is>
struct _impl_ebo_tuple<Tup, std::index_sequence<Is...>> : protected _impl_ebo_entry<Is, std::tuple_element_t<Is, Tup>>... {
    _impl_ebo_tuple() = default;

    _impl_ebo_tuple(std::tuple_element_t<Is, Tup> &&...ts) : _impl_ebo_entry<Is, std::tuple_element_t<Is, Tup>>(std::move(ts))... {
    }
};

template <class ...Ts>
struct ebo_tuple : private _impl_ebo_tuple<std::tuple<Ts...>, std::make_index_sequence<sizeof...(Ts)>> {
    explicit ebo_tuple() = default;

    explicit ebo_tuple(Ts &&...ts) : _impl_ebo_tuple<std::tuple<Ts...>, std::make_index_sequence<sizeof...(Ts)>>(std::move(ts)...) {
    }

    template <std::size_t I, class T = std::tuple_element_t<I, std::tuple<Ts...>>>
    T const &ebo_get() const {
        return static_cast<T const &>(static_cast<_impl_ebo_entry<I, T> const &>(*this));
    }

    template <std::size_t I, class T = std::tuple_element_t<I, std::tuple<Ts...>>>
    T &ebo_get() {
        return static_cast<T &>(static_cast<_impl_ebo_entry<I, T> &>(*this));
    }

    using ebo_base = ebo_tuple;
};

}
