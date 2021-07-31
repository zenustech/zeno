#pragma once

#include "common.h"

struct GLPrimitiveObject : zeno::IObjectClone<GLPrimitiveObject> {
    struct Attribute {
        std::vector<float> arr;
        int dim = 1;
    };

    std::vector<Attribute> m_attrs;
    size_t m_size;

    void clear() {
        m_attrs.clear();
        m_size = 0;
    }

    size_t add_attr(int dim) {
        m_attrs.emplace_back();
        auto back = m_attrs.back();
        back.arr.resize(m_size);
        back.dim = dim;
    }

    auto const &attrs() const {
        return m_attrs;
    }

    size_t size() const {
        return m_size;
    }

    size_t resize(size_t new_size) {
        m_size = new_size;
        for (int i = 0; i < m_attrs.size(); i++) {
            m_attrs[i].arr.resize(new_size);
        }
    }
};
