#pragma once


#include <tuple>
#include <type_traits>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _meta_tools_h {

constexpr auto tuple_push_front(auto const &t, auto const &tuple) {
    return std::tuple_cat(std::make_tuple(t), tuple);
}

constexpr auto tuple_push_back(auto const &t, auto const &tuple) {
    return std::tuple_cat(tuple, std::make_tuple(t));
}

template <class Tuple>
constexpr auto tuple_front(Tuple const &tuple) {
    return std::get<0>(tuple);
}

template <class Tuple>
constexpr auto tuple_back(Tuple const &tuple) {
    return std::get<std::tuple_size<Tuple>::value - 1>(tuple);
}

template <class Tuple>
constexpr auto tuple_pop_front(Tuple const &tuple) {
    return ([]<std::size_t ...Is> (auto const &tuple, std::index_sequence<Is...>) {
        return std::make_tuple(std::get<1 + Is>(tuple)...);
    })(tuple, std::make_index_sequence<std::tuple_size<Tuple>::value - 1>());
}

template <class Tuple>
constexpr auto tuple_pop_back(Tuple const &tuple) {
    return ([]<std::size_t ...Is> (auto const &tuple, std::index_sequence<Is...>) {
        return std::make_tuple(std::get<Is>(tuple)...);
    })(tuple, std::make_index_sequence<std::tuple_size<Tuple>::value - 1>());
}


template <int First, int Last, typename Lambda>
inline constexpr bool static_for(Lambda const &f) {
    if constexpr (First < Last) {
        if (f(std::integral_constant<int, First>{})) {
            return true;
        } else {
            return static_for<First + 1, Last>(f);
        }
    }
    return false;
}

template <class Variant, size_t I = 0>
inline Variant variant_from_index(size_t index) {
    if constexpr (I >= std::variant_size_v<Variant>)
        throw std::bad_variant_access{};
    else
        return index == 0 ? Variant{std::in_place_index<I>}
            : variant_from_index<Variant, I + 1>(index - 1);
}

}
}
ZENO_NAMESPACE_END
