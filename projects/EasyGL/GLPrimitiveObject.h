#pragma once

#include "common.h"
#include <zeno/core/IObject.h>

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject> {
    struct Attribute {
        std::vector<float> arr;
        int dim = 1;

        void _m_resize(size_t new_size) {
            arr.resize(dim * new_size);
        }
    };

    std::vector<Attribute> m_attrs;
    size_t m_size;

    void clear() {
        m_attrs.clear();
        m_size = 0;
    }

    int add_attr(int dim) {
        int id = m_attrs.size();
        m_attrs.emplace_back();
        auto &back = m_attrs.back();
        back.dim = dim;
        back._m_resize(dim * m_size);
        return id;
    }

    size_t nattrs() const {
        return m_attrs.size();
    }

    auto &attr(int i) {
        return m_attrs.at(i);
    }

    auto const &attr(int i) const {
        return m_attrs.at(i);
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t new_size) {
        m_size = new_size;
        for (int i = 0; i < m_attrs.size(); i++) {
            m_attrs[i]._m_resize(new_size);
        }
    }
};
