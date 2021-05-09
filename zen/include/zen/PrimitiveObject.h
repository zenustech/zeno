#pragma once

#include <zen/zen.h>
#include <zen/vec.h>
#include <variant>
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace zenbase {

using AttributeArray = std::variant<
    std::vector<zen::vec3f>, std::vector<float>>;

struct PrimitiveObject : zen::IObject {

    std::map<std::string, AttributeArray> m_attrs;
    size_t m_size{0};

    std::vector<int> particles;
    std::vector<zen::vec2i> lines;
    std::vector<zen::vec3i> triangles;
    std::vector<zen::vec4i> quads;

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
        return std::get<std::vector<T>>(m_attrs.at(name));
    }

    AttributeArray const &attr(std::string name) const {
        return m_attrs.at(name);
    }

    template <class T>
    bool attr_is(std::string name) const {
        return std::holds_alternative<std::vector<T>>(m_attrs.at(name));
    }

    size_t size() const {
        return m_size;
    }

    void resize(size_t size) {
        m_size = size;
        for (auto &[key, val]: m_attrs) {
            std::visit([&](auto &val) {
                val.resize(m_size);
            }, val);
        }
    }
};

}
