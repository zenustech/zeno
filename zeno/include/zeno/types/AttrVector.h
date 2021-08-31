#pragma once

#include <zeno/utils/vec.h>
#include <variant>
#include <vector>

namespace zeno {

template <class ValT>
struct AttrVector {
    using VariantType = std::variant<std::vector<vec3f>, std::vector<float>>;
    using ValueType = ValT;

    std::vector<ValT> values;
    std::map<std::string, VariantType> attrs;

    AttrVector() = default;
    AttrVector(std::vector<ValT> const &values_) : values(values_) {}
    AttrVector(std::vector<ValT> &&values_) : values(std::move(values_)) {}
    explicit AttrVector(size_t size) : values(size) {}

    operator auto &() {
        return values;
    }

    operator auto const &() const {
        return values;
    }

    void push_back(ValT const &t) {
        values.push_back(t);
    }

    void push_back(ValT &&t) {
        values.push_back(std::move(t));
    }

    template <class ...Ts>
    decltype(auto) emplace_back(Ts &&...ts) {
        values.emplace_back(std::forward<Ts>(ts)...);
    }

    template <class T>
    decltype(auto) at(T &&idx) const {
        return values.at(std::forward<T>(idx));
    }

    template <class T>
    decltype(auto) at(T &&idx) {
        return values.at(std::forward<T>(idx));
    }

    template <class T>
    decltype(auto) operator[](T &&idx) const {
        return values.operator[](std::forward<T>(idx));
    }

    template <class T>
    decltype(auto) operator[](T &&idx) {
        return values.operator[](std::forward<T>(idx));
    }

    template <class T>
    auto &add_attr(std::string const &name) {
        if (!has_attr(name))
            attrs[name] = std::vector<T>(size());
        return attr<T>(name);
    }

    template <class T>
    auto &add_attr(std::string const &name, T const &value) {
        if (!has_attr(name))
            attrs[name] = std::vector<T>(size(), value);
        return attr<T>(name);
    }

    template <class T>
    auto const &attr(std::string const &name) const {
        return std::get<std::vector<T>>(attrs.at(name));
    }

    template <class T>
    auto &attr(std::string const &name) {
        return std::get<std::vector<T>>(attrs.at(name));
    }

    auto const &attr(std::string const &name) const {
        return attrs.at(name);
    }

    auto &attr(std::string const &name) {
        return attrs.at(name);
    }

    bool has_attr(std::string const &name) const {
        return attrs.find(name) != attrs.end();
    }

    void erase_attr(std::string const &name) {
        attrs.erase(name);
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        return std::holds_alternative<std::vector<T>>(attrs.at(name));
    }

    void clear_attrs() {
        attrs.clear();
    }

    size_t size() const {
        return values.size();
    }

    void resize(size_t size) {
        values.resize(size);
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.resize(size); }, val);
        }
    }

    void clear() {
        values.clear();
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.clear(); }, val);
        }
    }
};

}
