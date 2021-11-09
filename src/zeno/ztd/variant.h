#pragma once


#include <zeno/ztd/functional.h>
#include <optional>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _H_variant {


template <class T, T ...Is>
using integral_variant = std::variant<std::integral_constant<T, Is>...>;


inline std::variant<std::true_type, std::false_type> make_bool_variant(bool cond) {
    if (cond) return std::true_type{};
    else return std::false_type{};
}


template <class Variant, size_t I = 0>
Variant variant_from_index(size_t index) {
    if constexpr (I >= std::variant_size_v<Variant>)
        throw std::bad_variant_access{};
    else
        return index == 0 ? Variant{std::in_place_index<I>}
            : variant_from_index<Variant, I + 1>(index - 1);
}


template <class T>
std::optional<T> try_get(auto const &var) {
    return std::visit(overloaded
    ( [] (T const &t) -> std::optional<T> { return std::make_optional(t); }
    , [] (auto const &) -> std::optional<T> { return std::nullopt; }
    ), var);
}


decltype(auto) match(auto &&var, auto &&...fs) {
    return std::visit(overloaded(
            std::forward<decltype(fs)>(fs)...),
        std::forward<decltype(var)>(var));
}


}
}
ZENO_NAMESPACE_END
