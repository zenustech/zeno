#pragma once

#include <vector>
#include <type_traits>
#include <memory>

namespace zeno {

template <class Map>
struct PolymorphicMap {
    using key_type = typename Map::key_type;
    using mapped_type = typename Map::mapped_type;

    using element_type = typename std::pointer_traits<mapped_type>::element_type;
    static_assert(std::is_polymorphic_v<element_type>);

    using value_type = typename Map::value_type;

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

    auto find(key_type const &key) const {
        return m_curr.find(key);
    }

    void clear() {
        m_curr.clear();
    }

    auto try_emplace(key_type const &key, mapped_type ptr) {
        return m_curr.try_emplace(key, std::move(ptr));
    }

    template <class Derived>
    std::vector<Derived *> values() const {
        static_assert(std::is_base_of_v<element_type, Derived>);
        std::vector<Derived *> ret;
        for (auto const &[key, ptr]: m_curr) {
            auto p = std::addressof(*ptr);
            if (auto q = dynamic_cast<Derived *>(p)) {
                ret.push_back(q);
            }
        }
        return ret;
    }

    std::vector<element_type *> values() const {
        std::vector<element_type *> ret;
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
            static_assert(std::is_base_of_v<element_type, Derived>);
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

}
