#pragma once

#include <vector>
#include <memory>
#include <zeno/memory.h>
#include <algorithm>
#include <string>
#include <cstdio>

struct Particles {
    std::vector<zeno::copiable_unique_ptr<std::vector<float>>> m_arrs;
    std::vector<std::string> m_chnames;
    float m_size;

    size_t nchannels() const {
        return m_arrs.size();
    }

    std::string channel_name(size_t i) const {
        return m_chnames[i];
    }

    void set_channel_name(size_t i, std::string const &name) {
        m_chnames[i] = name;
    }

    size_t chid_of_name(std::string const &name) const {
        auto it = std::find(m_chnames.begin(), m_chnames.end(), name);
        return it - m_chnames.begin();
    }

    std::vector<float> &channel(size_t i) const {
        return *m_arrs[i].get();
    }

    void set_nchannels(size_t n) {
        size_t m = m_arrs.size();
        m_arrs.resize(n);
        m_chnames.resize(n);
        for (size_t i = m; i < n; i++) {
            m_arrs[i] = std::make_unique<std::vector<float>>(m_size);
        }
        for (size_t i = m; i < n; i++) {
            char buf[233];
            sprintf(buf, "ch%d", i);
            m_chnames[i] = buf;
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
