#pragma once

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace zeno {

namespace __range_to_initializer_list {

    constexpr size_t DEFAULT_MAX_LENGTH = 128;

    template <typename V> struct backingValue { static V value; };
    template <typename V> V backingValue<V>::value;

    template <typename V, typename... Vcount> struct backingList { static std::initializer_list<V> list; };
    template <typename V, typename... Vcount>
    std::initializer_list<V> backingList<V, Vcount...>::list = {(Vcount)backingValue<V>::value...};

    template <size_t maxLength, typename It, typename V = typename It::value_type, typename... Vcount>
    static typename std::enable_if< sizeof...(Vcount) >= maxLength,
    std::initializer_list<V> >::type generate_n(It begin, It end, It current)
    {
        throw std::length_error("More than maxLength elements in range.");
    }

    template <size_t maxLength = DEFAULT_MAX_LENGTH, typename It, typename V = typename It::value_type, typename... Vcount>
    static typename std::enable_if< sizeof...(Vcount) < maxLength,
    std::initializer_list<V> >::type generate_n(It begin, It end, It current)
    {
        if (current != end)
            return generate_n<maxLength, It, V, V, Vcount...>(begin, end, ++current);

        current = begin;
        for (auto it = backingList<V,Vcount...>::list.begin();
             it != backingList<V,Vcount...>::list.end();
             ++current, ++it)
            *const_cast<V*>(&*it) = *current;

        return backingList<V,Vcount...>::list;
    }

}

template <typename It>
std::initializer_list<typename It::value_type> to_initializer_list(It begin, It end)
{
    return __range_to_initializer_list::generate_n(begin, end, begin);
}


template <typename V>
auto to_initializer_list(V const &v)
{
    return to_initializer_list(v.begin(), v.end());
}

}
