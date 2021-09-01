#pragma once

#include <zeno/utils/vec.h>
#include <variant>
#include <vector>
#include <map>

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

    decltype(auto) begin() const {
        return values.begin();
    }

    decltype(auto) end() const {
        return values.end();
    }

    decltype(auto) data() const {
        return values.data();
    }

    decltype(auto) begin() {
        return values.begin();
    }

    decltype(auto) end() {
        return values.end();
    }

    decltype(auto) data() {
        return values.data();
    }

    decltype(auto) at(size_t idx) const {
        return values.at(idx);
    }

    decltype(auto) at(size_t idx) {
        return values.at(idx);
    }

    void push_back(ValT const &t) {
        values.push_back(t);
    }

    void push_back(ValT &&t) {
        values.push_back(std::move(t));
    }

    void update() {
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.resize(size); }, val);
        }
    }

    decltype(auto) operator[](size_t idx) const {
        return values[idx];
    }

    decltype(auto) operator[](size_t idx) {
        return values[idx];
    }

    auto const *operator->() const {
        return &values;
    }

    auto *operator->() {
        return &values;
    }

    operator auto const &() const {
        return values;
    }

    operator auto &() {
        return values;
    }

    template <class F>
    void foreach_attr(F const &f) const {
        for (auto const &[key, arr]: attrs) {
            std::visit([&] (auto &arr) {
                f(key, arr);
            }, arr);
        }
    }

    template <class F>
    void foreach_attr(F const &f) {
        for (auto &[key, arr]: attrs) {
            std::visit([&] (auto &arr) {
                f(key, arr);
            }, arr);
        }
    }

    size_t num_attrs() const {
        return attrs.size();
    }

    auto attr_keys() const {
        std::vector<std::string> keys;
        for (auto const &[key, val]: attrs) {
            keys.push_back(key);
        }
        return keys;
    }

    template <class ...Ts>
    decltype(auto) emplace_back(Ts &&...ts) {
        values.emplace_back(std::forward<Ts>(ts)...);
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
