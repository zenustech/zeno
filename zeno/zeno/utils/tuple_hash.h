#pragma once

#include <type_traits>
#include <functional>
#include <array>
#include <tuple>

namespace zeno {

struct tuple_hash {
    static constexpr std::array<std::size_t, 3> magics = {
        2718281828, 3141592653, 1618033989,
    };

    template <class T, std::size_t ...Is>
    std::size_t _call_helper(T const &t, std::index_sequence<Is...>) const {
        return ((std::hash<std::tuple_element_t<Is, T>>{}(std::get<Is>(t)) * std::get<Is>(magics)) ^ ...);
    }

    template <class T, std::enable_if_t<std::tuple_size<T>::value != 0, int> = 0>
    std::size_t operator()(T const &t) const {
        static_assert(std::tuple_size_v<T> <= std::size(magics), "maybe you need to add more magic numbers");
        return _call_helper(t, std::make_index_sequence<std::tuple_size<T>::value>{});
    }
};

template <class Op>
struct tuple_operator {
    Op op{};

    template <class T, std::size_t ...Is>
    bool _call_helper(T const &t1, T const &t2, std::index_sequence<Is...>) const {
        return op(std::tie(std::get<Is>(t1)...), std::tie(std::get<Is>(t2)...));
    }

    template <class T, std::enable_if_t<std::tuple_size<T>::value != 0, int> = 0>
    bool operator()(T const &t1, T const &t2) const {
        return _call_helper(t1, t2, std::make_index_sequence<std::tuple_size<T>::value>{});
    }
};

using tuple_less = tuple_operator<std::less<>>;
using tuple_equal = tuple_operator<std::equal_to<>>;

// Usage example:
// std::unordered_map<zeno::vec3i, ValueType, tuple_hash, tuple_equal>
// std::map<zeno::vec3i, ValueType, tuple_less>

}
