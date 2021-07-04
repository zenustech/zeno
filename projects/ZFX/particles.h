#pragma once

#include <vector>
#include <memory>
#include <zeno/memory.h>

struct Particles {
    std::vector<zeno::copiable_unique_ptr<std::vector<float>>> m_arrs;
    float m_size;

    size_t nchannels() const {
        return m_arrs.size();
    }

    std::vector<float> *channel(size_t i) const {
        return m_arrs[i].get();
    }

    void set_nchannels(size_t n) {
        size_t m = m_arrs.size();
        m_arrs.resize(n);
        for (size_t i = m; i < n; i++) {
            m_arrs[i] = std::make_unique<std::vector<float>>(m_size);
        }
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t n) {
        m_size = n;
        for (auto const &arr: m_arrs) {
            arr->resize(n);
        }
    }
};
