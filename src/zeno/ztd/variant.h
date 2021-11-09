#pragma once


#include <zeno/ztd/functional.h>
#include <optional>
#include <variant>


ZENO_NAMESPACE_BEGIN
namespace ztd {
inline namespace _H_variant {


template <class T>
std::optional<T> try_get(auto const &var) {
    return std::visit(match
    ( [] (T const &t) -> std::optional<T> { return std::make_optional(t); }
    , [] (auto const &) -> std::optional<T> { return std::nullopt; }
    ), var);
}


template <class T>
decltype(auto) match_visit(auto const &var, auto &&...fs) {
    return std::visit(match(std::forward<decltype(fs)>(fs)...), var);
}


}
}
ZENO_NAMESPACE_END
