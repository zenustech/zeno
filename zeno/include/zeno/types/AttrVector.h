#pragma once

#include <zeno/utils/vec.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/type_traits.h>
#include <variant>
#include <vector>
#include <map>

namespace zeno {

using AttrAcceptAll = std::variant
    < vec3f
    , float
    , vec3i
    , int
    , vec2f
    , vec2i
    , vec4f
    , vec4i
    >;


template <class ValT>
struct AttrVector {
    using AttrVectorVariant = std::variant
        < std::vector<vec3f>
        , std::vector<float>
        , std::vector<vec3i>
        , std::vector<int>
        , std::vector<vec2f>
        , std::vector<vec2i>
        , std::vector<vec4f>
        , std::vector<vec4i>
        >;

    using value_type = ValT;
    using BaseVector = std::vector<ValT>;
    using size_type = typename BaseVector::size_type;
    using pointer = typename BaseVector::pointer;
    using reference = typename BaseVector::reference;
    using const_pointer = typename BaseVector::const_pointer;
    using const_reference = typename BaseVector::const_reference;
    using iterator = typename BaseVector::iterator;
    using const_iterator = typename BaseVector::const_iterator;

    inline static const std::string kpos = "pos"; 

    BaseVector values;
    std::map<std::string, AttrVectorVariant> attrs;

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

    //void _ensure_update() const {
        //if (!attrs.empty()) {
            //const_cast<AttrVector *>(this)->update();
        //}
    //}

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

    template <class Accept = std::variant<vec3f, float>, class F>
    void attr_visit(std::string const &name, F const &f) const {
        if (name == "pos") {
            f(values);
            return;
        }
        auto it = attrs.find(name);
        if (it == attrs.end())
            throw makeError<KeyError>(name, "attribute name of primitive");
        std::visit([&] (auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (variant_contains<T, Accept>::value) {
                f(arr);
            }
        }, it->second);
    }

    template <class Accept = std::variant<vec3f, float>, class F>
    void attr_visit(std::string const &name, F const &f) {
        if constexpr (variant_contains<ValT, Accept>::value) {
            if (name == "pos") {
                f(values);
                return;
            }
        }
        auto it = attrs.find(name);
        if (it == attrs.end())
            throw makeError<KeyError>(name, "attribute name of primitive");
        std::visit([&] (auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            if constexpr (variant_contains<T, Accept>::value) {
                f(arr);
            }
        }, it->second);
    }

    template <class Accept = std::variant<vec3f, float>, class F>
    void foreach_attr(F &&f) const {
        for (auto const &[key, arr]: attrs) {
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        }
    }

    template <class Accept = std::variant<vec3f, float>, class F>
    void foreach_attr(F &&f) {
        for (auto &[key, arr]: attrs) {
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        }
    }

    template <class Accept = std::variant<vec3f, float>, class F>
    void forall_attr(F &&f) const {
        f(kpos, values);
        for (auto const &[key, arr]: attrs) {
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        }
    }

    template <class Accept = std::variant<vec3f, float>, class F>
    void forall_attr(F &&f) {
        f(kpos, values);
        for (auto &[key, arr]: attrs) {
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        }
    }

    /*
#ifdef ZENO_PARALLEL_STL
    template <class Accept = std::variant<vec3f, float>, class F, class Pol>
    void foreach_attr(Pol pol, F &&f) const {
        std::for_each(pol, attrs.begin(), attrs.end(), [&] (auto &kv) {
            auto const &[key, arr] = kv;
            auto &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        });
    }

    template <class Accept = std::variant<vec3f, float>, class F, class Pol>
    void foreach_attr(Pol pol, F &&f) {
        std::for_each(pol, attrs.begin(), attrs.end(), [&] (auto &kv) {
            auto &[key, arr] = kv;
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        });
    }

    template <class Accept = std::variant<vec3f, float>, class F, class Pol>
    void forall_attr(Pol pol, F &&f) const {
        const std::string kpos = "pos";
        f(kpos, values);
        std::for_each(pol, attrs.begin(), attrs.end(), [&] (auto &kv) {
            auto const &[key, arr] = kv;
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        });
    }

    template <class Accept = std::variant<vec3f, float>, class F, class Pol>
    void forall_attr(Pol pol, F &&f) {
        const std::string kpos = "pos";
        f(kpos, values);
        std::for_each(pol, attrs.begin(), attrs.end(), [&] (auto &kv) {
            auto &[key, arr] = kv;
            auto const &k = key;
            std::visit([&] (auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                if constexpr (variant_contains<T, Accept>::value) {
                    f(k, arr);
                }
            }, arr);
        });
    }
#endif
    */

    template <class Accept = std::variant<vec3f, float>>
    size_t num_attrs() const {
        if constexpr (std::is_same_v<Accept, AttrAcceptAll>) {
            return attrs.size();
        } else {
            size_t count = 0;
            foreach_attr<Accept>([&] (auto const &key, auto const &arr) {
                count++;
            });
            return count;
        }
    }

    template <class Accept = std::variant<vec3f, float>>
    auto attr_keys() const {
        std::vector<std::string> keys;
        foreach_attr<Accept>([&] (auto const &key, auto const &arr) {
            keys.push_back(key);
        });
        return keys;
    }

    template <class ...Ts>
    decltype(auto) emplace_back(Ts &&...ts) {
        values.emplace_back(std::forward<Ts>(ts)...);
    }

    template <class T>
    auto &add_attr(std::string const &name) {
        if (!attr_is<T>(name))
            attrs[name] = std::vector<T>(size());
        return attr<T>(name);
    }

    // deprecated:
    template <class T>
    auto &add_attr(std::string const &name, T const &val) {
        if (!attr_is<T>(name))
            attrs[name] = std::vector<T>(size(), val);
        return attr<T>(name);
    }

    //template <class T>
    //auto &add_attr(std::string const &name, T const &value) {
        //if (!attr_is<T>(name))
            //attrs[name] = std::vector<T>(size(), value);
        //return attr<T>(name);
    //}

    template <class T>
    auto const &attr(std::string const &name) const {
        if (name == "pos") {
            if constexpr (!std::is_same_v<T, ValT>) {
                throw makeError<TypeError>(typeid(T), typeid(ValT), "type of primitive attribute pos");
            } else {
                return values;
            }
        }
        auto const &arr = attr(name);
        if (!std::holds_alternative<std::vector<T>>(arr))
            throw makeError<TypeError>(typeid(T), std::visit([&] (auto const &t) -> std::type_info const & { return typeid(std::decay_t<decltype(t[0])>); }, arr), "type of primitive attribute " + name);
        return std::get<std::vector<T>>(arr);
    }

    template <class T>
    auto &attr(std::string const &name) {
        if (name == "pos") {
            if constexpr (!std::is_same_v<T, ValT>) {
                throw makeError<TypeError>(typeid(T), typeid(ValT), "type of primitive attribute pos");
            } else {
                return values;
            }
        }
        auto &arr = attr(name);
        if (!std::holds_alternative<std::vector<T>>(arr))
            throw makeError<TypeError>(typeid(T), std::visit([&] (auto const &t) -> std::type_info const & { return typeid(std::decay_t<decltype(t[0])>); }, arr), "type of primitive attribute " + name);
        return std::get<std::vector<T>>(arr);
    }

    // deprecated:
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
            throw makeError<KeyError>(name, "attribute name of primitive");
        return it->second;
    }

    // deprecated:
    auto &attr(std::string const &name) {
        //_ensure_update();
        auto it = attrs.find(name);
        if (it == attrs.end())
            throw makeError<KeyError>(name, "attribute name of primitive");
        return it->second;
    }

    bool has_attr(std::string const &name) const {
        if (name == "pos") return true;
        return attrs.find(name) != attrs.end();
    }

    void erase_attr(std::string const &name) {
        attrs.erase(name);
    }

    template <class T>
    bool attr_is(std::string const &name) const {
        if (name == "pos") return std::is_same_v<T, ValT>;
        auto it = attrs.find(name);
        return it != attrs.end() && std::holds_alternative<std::vector<T>>(it->second);
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
