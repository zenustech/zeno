#pragma once

#include <zeno/zeno.h>
#include "Array.h"
#include <string>

struct Particles : zeno::IObject {
    std::map<std::string, VariantArray> m_attrs;
    size_t m_size = 0;

    template <class F>
    void foreach_attr(F const &f) {
        for (auto &[key, arr]: m_attrs) {
            std::visit([&](auto &arr) {
                f(key, arr);
            }, arr);
        }
    }

    template <class F>
    void visit_attr(std::string const &key, F const &f) {
        std::visit([&](auto &arr) {
            f(arr);
        }, m_attrs.at(key));
    }

    template <class T, size_t N>
    auto &attr(std::string const &key) {
        return std::get<Array<T, N>>(m_attrs.at(key));
    }

    template <class T, size_t N>
    auto &add_attr(std::string const &key) {
        if (auto it = m_attrs.find(key); it != m_attrs.end()) {
            return std::get<Array<T, N>>(it->second);
        }
        m_attrs[key] = Array<T, N>();
        auto &ret = attr<T, N>(key);
        ret.resize(m_size);
        return ret;
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        foreach_attr([&] (auto const &key, auto &arr) {
            arr.resize(size);
        });
        m_size = size;
    }
};
