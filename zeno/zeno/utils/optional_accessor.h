#pragma once

#include <string>

template <class T, class Enabled, class AttVec>
static auto optional_attribute_accessor(AttVec &attvec, std::string const &name, Enabled enabled = {}, T defl = {}) {
    if constexpr (enabled) {
        auto &arr = attvec.template attr<T>(name);
        return [&] (std::size_t i) -> decltype(auto) {
            return arr[i];
        };
    } else {
        return [defl] (std::size_t) {
            return defl;
        };
    }
}
