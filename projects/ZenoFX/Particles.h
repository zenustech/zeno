#pragma once

#include <zeno/zeno.h>
#include "Array.h"
#include <string>

struct Particles : zeno::IObject {
    std::map<std::string, VariantArray> attrs;

    template <class F>
    void foreach_attr(F const &f) {
        for (auto &[key, arr]: attrs) {
            std::visit([&key, &f](auto &arr) {
                f(key, arr);
            }, arr);
        }
    }

    template <class T, size_t N>
    auto &attr(std::string const &key) {
        return std::get<Array<T, N>>(attrs.at(key));
    }

    template <class T, size_t N>
    auto &add_attr(std::string const &key) {
        if (auto it = attrs.find(key); it != attrs.end()) {
            return std::get<Array<T, N>>(it->second);
        }
        attrs[key] = Array<T, N>();
        return attr<T, N>(key);
    }
};
