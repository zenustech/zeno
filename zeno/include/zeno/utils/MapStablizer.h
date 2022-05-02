#pragma once

#include <utility>
#include <zeno/utils/scope_exit.h>

namespace zeno {

template <class Map>
struct MapStablizer {
    using key_type = typename Map::key_type;
    using mapped_type = typename Map::mapped_type;
    using iterator = typename Map::iterator;
    using const_iterator = typename Map::const_iterator;

    Map m_curr, m_next;

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

    template <class T = void>
    decltype(auto) values() const {
        if constexpr (std::is_void_v<T>)
            return m_curr.values();
        else
            return m_curr.template values<T>();
    }

    template <class T = void>
    decltype(auto) pairs() const {
        if constexpr (std::is_void_v<T>)
            return m_curr.values();
        else
            return m_curr.template values<T>();
    }

    template <class T>
    std::size_t size() const {
        return m_curr.template size<T>();
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

    struct InsertPass : scope_finalizer<InsertPass> {
        MapStablizer &that;

        explicit InsertPass(MapStablizer &that_) : that(that_) {}

        bool may_emplace(key_type const &key) {
            auto it = that.m_curr.find(key);
            if (it != that.m_curr.end()) {
                that.m_next.try_emplace(key, std::move(const_cast<mapped_type &>(it->second)));
                return false;
            }
            return true;
        }

        template <class ...Args>
        auto try_emplace(key_type const &key, Args &&...args) {
            return that.m_next.try_emplace(key, std::forward<Args>(args)...);
        }

        void _scope_finalize() {
            std::swap(that.m_curr, that.m_next);
            that.m_next.clear();
        }
    };

    InsertPass insertPass() {
        return InsertPass(*this);
    }
};

}
