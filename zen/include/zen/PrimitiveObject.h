#pragma once

#include <zen/zen.h>
#include <variant>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <map>

namespace zenbase {

struct PrimitiveObject : zen::IObject {

    using AttributeArray = std::variant<
        std::vector<std::array<float, 3>>, std::vector<float>>;

    std::map<std::string, AttributeArray> m_attrs;
    size_t m_size{0};

    template <class T>
    void add_attr(std::string name) {
        m_attrs[name] = std::vector<T>(m_size);
    }

    template <class T>
    std::vector<T> attr(std::string name) const {
        return m_attrs.at(name);
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        m_size = size;
        for (auto &[key, val]: m_attrs) {
            if (0) {
#define _PER_ALTER(T...) \
            } else if (std::holds_alternative<std::vector<T>>(val)) { \
                std::get<std::vector<T>>(val).resize(m_size);
            _PER_ALTER(std::array<float, 3>)
            _PER_ALTER(float)
#undef _PER_ALTER
            }
        }
    }
};

}
