#pragma once

#include <zeno/utils/vec.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/type_traits.h>
#include <variant>
#include <vector>
#include <map>

namespace zeno {

template <class ValT>
struct AttrVector {
    using VariantType = std::variant
        < std::vector<vec3f>
        , std::vector<float>
        , std::vector<vec3i>
        , std::vector<int>
        >;

    using value_type = ValT;
    using iterator = typename std::vector<ValT>::iterator;
    using const_iterator = typename std::vector<ValT>::const_iterator;

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

    void _ensure_update() const {
        if (!attrs.empty()) {
            const_cast<AttrVector *>(this)->update();
        }
    }

    void update() {
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.resize(this->size()); }, val);
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

    template <class Accept = std::tuple<vec3f, float>, class F>
    void attr_visit(std::string const &name, F const &f) const {
        std::visit([&] (auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (tuple_contains<T, Accept>::value) {
                f(arr);
            }
        }, attr(name));
    }

    template <class Accept = std::tuple<vec3f, float>, class F>
    void attr_visit(std::string const &name, F const &f) {
        std::visit([&] (auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (tuple_contains<T, Accept>::value) {
                f(arr);
            }
        }, attr(name));
    }

    template <class Accept = std::tuple<vec3f, float>, class F>
    void foreach_attr(F &&f) const {
        for (auto const &[key, arr]: attrs) {
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (tuple_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        }
    }

    template <class Accept = std::tuple<vec3f, float>, class F>
    void foreach_attr(F &&f) {
        for (auto &[key, arr]: attrs) {
            auto &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (tuple_contains<T, Accept>::value) {
                    f(k, arr);
                }
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
        return std::get<std::vector<T>>(attr(name));
    }

    template <class T>
    auto &attr(std::string const &name) {
        return std::get<std::vector<T>>(attr(name));
    }

    auto const &attr(std::string const &name) const {
        //this causes bug in primitive clip
        //reason: in primitiveClip, we will emplace back to attr by
        //means like attr<T>.emplace_back(val)
        //suppose "pos" = {}
        //        "clr" = {}
        //attr<vec3f>("clr").emplace_back(val)
        //attr<vec3f>("pos").emplace_back(val)<---this will resize "clr" to zero first and then push_back to "pos"
        //_ensure_update();
        auto it = attrs.find(name);
        if (it == attrs.end())
            throw makeError<KeyError>(name, "attribute", "PrimitiveObject::attr");
        return it->second;
    }

    auto &attr(std::string const &name) {
        //_ensure_update();
        auto it = attrs.find(name);
        if (it == attrs.end())
            throw makeError<KeyError>(name, "attribute", "PrimitiveObject::attr");
        return it->second;
    }

    bool has_attr(std::string const &name) const {
        return attrs.find(name) != attrs.end();
    }

    void erase_attr(std::string const &name) {
        attrs.erase(name);
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        return std::holds_alternative<std::vector<T>>(attr(name));
    }

    void clear_attrs() {
        attrs.clear();
    }

    size_t size() const {
        return values.size();
    }

    void reserve(size_t size) {
        values.reserve(size);
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.reserve(size); }, val);
        }
    }

    void shrink_to_fit() {
        values.shrink_to_fit();
        for (auto &[key, val] : attrs) {
            std::visit([&](auto &val) { val.shrink_to_fit(); }, val);
        }
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
