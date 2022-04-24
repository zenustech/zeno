#pragma once

#include <utility>
#include <unordered_map>
#include <zeno/utils/scope_exit.h>

namespace zeno {

template <class Key, class Val, class Map = std::unordered_map<Key, Val>>
struct MapStablizer {
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

    template <class T>
    std::size_t size() const {
        return m_curr.template size<T>();
    }

    std::size_t size() const {
        return m_curr.size();
    }

    auto find(Key const &key) const {
        return m_curr.find(key);
    }

    struct InsertPass : scope_finalizer<InsertPass> {
        MapStablizer &that;

        explicit InsertPass(MapStablizer &that_) : that(that_) {}

        template <class ...Args>
        bool try_emplace(Key const &key, Args &&...args) {
            auto [it, succ] = that.m_next.try_emplace(key, std::forward<Args>(args)...);
            return succ;
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
