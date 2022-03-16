#pragma once

#include <zeno/utils/variantswitch.h>
#include <algorithm>
#include <string>

template <class Variant, bool HasMono = false, class Name, class Table, std::size_t N>
Variant string_variant(std::string name, Table (&table)[N]) {
    std::size_t index{std::find(std::begin(table), std::end(table), name) - std::begin(table)};
    return index_switch<std::variant_size_v<Variant>, HasMono>(index, [] (auto index) {
        return Variant{std::in_place_index<index.value>};
    });
}

template <class Enum = std::size_t, bool HasMono = false, class Name, class Table, std::size_t N>
Enum string_enum(std::string name, Table (&table)[N]) {
    std::size_t index{std::find(std::begin(table), std::end(table), name) - std::begin(table)};
    if constexpr (HasMono)
        if (index == std::size(table))
            throw std::bad_variant_access{};
    if constexpr (std::is_enum_v<Enum>)
        return std::size_t{std::underlying_type_t<Enum>(index)};
    else
        return index;
}

