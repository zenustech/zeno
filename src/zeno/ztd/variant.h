#pragma once


#include <variant>
#include <utility>
#include <type_traits>


template <class T>
auto make_monovariant_if(auto &&cond, T x) {
    std::variant<std::monostate, T> ret;
    if (cond) ret = std::move(x);
    return ret;
}


template <class T>
concept not_monostate = !std::is_same_v<std::monostate, std::remove_cvref_t<T>>;
