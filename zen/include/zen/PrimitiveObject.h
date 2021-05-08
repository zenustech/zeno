#pragma once

#include <zen/zen.h>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <variant>
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace zenbase {

using AttributeArray = std::variant<
    std::vector<glm::vec3>, std::vector<float>>;

struct PrimitiveObject : zen::IObject {

    std::map<std::string, AttributeArray> m_attrs;
    size_t m_size{0};
    std::vector<int> particles;
    std::vector<glm::ivec2> lines;
    std::vector<glm::ivec3> triangles;
    std::vector<glm::ivec4> quads;
    template <class T>
    void add_attr(std::string name) {
        m_attrs[name] = std::vector<T>(m_size);
    }

    template <class T>
    std::vector<T> &attr(std::string name) {
        return std::get<std::vector<T>>(m_attrs.at(name));
    }

    AttributeArray &attr(std::string name) {
        return m_attrs.at(name);
    }

    template <class T>
    std::vector<T> const &attr(std::string name) const {
        return m_attrs.at(name);
    }

    AttributeArray const &attr(std::string name) const {
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
            _PER_ALTER(glm::vec3)
            _PER_ALTER(float)
#undef _PER_ALTER
            }
        }
    }
};

}
