#pragma once

#include <variant>
#include <type_traits>
#include <typeinfo>
#include <algorithm>

namespace zeno {
inline namespace variantswitch_h {

static std::variant<std::false_type, std::true_type> boolean_variant(bool b) {
    if (b) return std::true_type{};
    else return std::false_type{};
}

using boolean_variant_t = std::variant<std::false_type, std::true_type>;

template <class Func>
decltype(auto) boolean_switch(bool b, Func &&func) {
    return std::visit(std::forward<Func>(func), boolean_variant(b));
}

struct index_variant_monostate {
    static inline constexpr size_t value = static_cast<size_t>(-1);
};

namespace index_variant_details {
    template <class Ret, bool HasMono, std::size_t I, std::size_t N>
    Ret helper_impl(std::size_t i) {
        if constexpr (I >= N) {
            if constexpr (HasMono) {
                return index_variant_monostate{};
            } else {
                throw std::bad_variant_access{};
            }
        } else {
            if (i == I) {
                return std::integral_constant<std::size_t, I>{};
            } else {
                return helper_impl<Ret, HasMono, I + 1, N>(i);
            }
        }
    }

    template <std::size_t N, bool HasMono, std::size_t ...Is>
    auto helper_call(std::size_t i, std::index_sequence<Is...>) {
        using Ret = std::conditional_t<HasMono
            , std::variant<index_variant_monostate, std::integral_constant<std::size_t, Is>...>
            , std::variant<std::integral_constant<std::size_t, Is>...>
            >;
        return helper_impl<Ret, HasMono, 0, N>(i);
    }
}

template <std::size_t N, bool HasMono = false>
using index_variant_t = decltype(index_variant_details::helper_call<N, HasMono>(0, std::make_index_sequence<N>{}));

template <std::size_t N, bool HasMono = false>
static auto index_variant(std::size_t i) {
    return index_variant_details::helper_call<N, HasMono>(i, std::make_index_sequence<N>{});
}

template <std::size_t N, bool HasMono = false, class Func>
decltype(auto) index_switch(std::size_t i, Func &&func) {
    return std::visit(std::forward<Func>(func), index_variant<N, HasMono>(i));
}

template <class Variant, class Enum>
Variant enum_variant(Enum e) {
    std::size_t index;
    if constexpr (std::is_enum_v<Enum>)
        index = std::size_t{std::underlying_type_t<Enum>(e)};
    else
        index = std::size_t{e};
    return index_switch<std::variant_size_v<Variant>>(index, [] (auto index) {
        return Variant{std::in_place_index<index.value>};
    });
}

template <class To, class From>
To variant_cast(From from) {
    static_assert(std::variant_size_v<From> == std::variant_size_v<To>);
    return index_switch<std::variant_size_v<From>>(from.index(), [] (auto index) {
        return To{std::in_place_index<index.value>};
    });
}

template <class Variant>
auto const &typeid_of_variant(Variant const &var) {
    return std::visit([&] (auto const &val) -> auto const & {
        return typeid(std::decay_t<decltype(val)>);
    }, var);
}

template <class Variant, class T>
struct variant_index {
};

template <class T, class ...Ts>
struct variant_index<std::variant<T, Ts...>, T> : std::integral_constant<std::size_t, 0> {
};

template <class T, class T0, class ...Ts>
struct variant_index<std::variant<T0, Ts...>, T> : variant_index<std::variant<Ts...>, T> {
};

template <class Enum, class Variant, class T>
struct variant_enum : std::integral_constant<Enum, Enum{std::underlying_type_t<Enum>(variant_index<Variant, T>::value)}> {
};

#if 0
template <class Variant, bool HasMono = false, class Table, std::size_t N>
Variant string_variant(std::string name, Table const (&table)[N]) {
    static_assert(N == std::variant_size_v<Variant>);
    std::size_t index{std::find(std::begin(table), std::end(table), name) - std::begin(table)};
    return index_switch<std::variant_size_v<Variant>, HasMono>(index, [] (auto index) {
        return Variant{std::in_place_index<index.value>};
    });
}

template <class Enum = std::size_t, bool HasMono = false, class Table, std::size_t N>
Enum string_enum(std::string name, Table const (&table)[N]) {
    std::size_t index{std::find(std::begin(table), std::end(table), name) - std::begin(table)};
    if constexpr (HasMono)
        if (index == std::size(table))
            throw std::bad_variant_access{};
    if constexpr (std::is_enum_v<Enum>)
        return std::size_t{std::underlying_type_t<Enum>(index)};
    else
        return index;
}
#endif

}
}
