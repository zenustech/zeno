#pragma once

#include <vector>
#include <unordered_map>
#include <type_traits>
#include <memory>

namespace zeno {

template <class Key, class Ptr, class Map = std::unordered_map<Key, Ptr>>
struct PolymorphicMap {
    using Base = typename std::pointer_traits<Ptr>::element_type;
    static_assert(std::is_polymorphic_v<Base>);

    Map m_curr;

    auto begin() const {
        return m_curr.begin();
    }

    auto end() const {
        return m_curr.end();
    }

    auto begin() {
        return m_curr.begin();
    }

    auto end() {
        return m_curr.end();
    }

    std::size_t size() const {
        return m_curr.size();
    }

    auto find(Key const &key) const {
        return m_curr.find(key);
    }

    bool try_emplace(Key const &key, Ptr ptr) {
        auto [it, succ] = m_curr.try_emplace(key, std::move(ptr));
        return succ;
    }

    template <class Derived>
    std::vector<Derived *> values() const {
        static_assert(std::is_base_of_v<Base, Derived>);
        std::vector<Derived *> ret;
        for (auto const &[key, ptr]: m_curr) {
            auto p = std::addressof(*ptr);
            if (auto q = dynamic_cast<Derived *>(p)) {
                ret.push_back(q);
            }
        }
        return ret;
    }

    std::vector<Base *> values() const {
        std::vector<Base *> ret;
        ret.reserve(m_curr.size());
        for (auto const &[key, ptr]: m_curr) {
            auto p = std::addressof(*ptr);
            ret.push_back(p);
        }
        return ret;
    }

    template <class Derived = void>
    std::size_t size() const {
        if constexpr (std::is_void_v<Derived>) {
            return m_curr.size();
        } else {
            static_assert(std::is_base_of_v<Base, Derived>);
            std::size_t ret = 0;
            for (auto const &[key, ptr]: m_curr) {
                auto p = std::addressof(*ptr);
                if (dynamic_cast<Derived *>(p)) {
                    ++ret;
                }
            }
            return ret;
        }
    }
};

};

}
