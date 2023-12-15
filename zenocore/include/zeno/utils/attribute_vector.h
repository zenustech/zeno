#pragma once

#include "generic_vector.h"
#include <string>
#include <map>

namespace zeno {

struct attribute_vector {
    std::map<std::string, comp_vector_storage> m_arr;

    bool has_attr(std::string const &name) const {
        return m_arr.find(name) != m_arr.end();
    }

    void add_attr(std::string const &name, dtype_e dtype, std::size_t dim) {
        m_arr.try_emplace(name, dtype, dim);
    }

    template <class T>
    comp_vector_viewer<T> add_attr(std::string const &name) {
        auto [it, succ] = m_arr.try_emplace(name, std::in_place_type<T>);
        return it->second.template view<T>();
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        return m_arr.at(name).template dtype_is<T>();
    }

    template <class T>
    comp_vector_viewer<T> attr(std::string const &name) const {
        return m_arr.at(name).template view<T>();
    }

    template <class Func>
    void attr_visit(std::string const &name, Func const &func) const {
        m_arr.at(name).visit_view([&] (auto const &v) {
            func(v);
        });
    }

    template <class Func>
    void foreach_attr(Func const &func) const {
        for (auto const &p: m_arr) {
            p.second.visit_view([&] (auto const &v) {
                func(p.first, v);
            });
        }
    }
};

}
