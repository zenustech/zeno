#pragma once

#include "HashMap.h"

namespace fdb {

template <class K, class T>
struct HashListMap {
    static_assert(std::is_constructible<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    static_assert(std::is_constructible<K>::value);
    static_assert(std::is_trivially_copy_constructible<K>::value);
    static_assert(std::is_trivially_copy_assignable<K>::value);
    static_assert(std::is_trivially_move_constructible<K>::value);
    static_assert(std::is_trivially_move_assignable<K>::value);
    static_assert(std::is_trivially_destructible<K>::value);

    struct Chunk {
        T m_data{};
        Chunk *m_next{nullptr};
    };

    struct Leaf {
        // HashMap will zero initialize these:
        Chunk *m_head;
        int m_lock;

        inline FDB_DEVICE T &append() {
            auto *chunk = (Chunk *)dynamic_allocate(sizeof(Chunk));
            new (chunk) Chunk();
            auto *old_head = atomic_swap(&m_head, chunk);
            chunk->m_next = old_head;
            return chunk->m_data;
        }

        inline FDB_DEVICE T &__init_head_nonatomic() {
            auto *chunk = (Chunk *)dynamic_allocate(sizeof(Chunk));
            new (&chunk->m_data) T();
            atomic_store(&m_head, chunk);
            chunk->m_next = nullptr;
            return chunk->m_data;
        }

        inline FDB_DEVICE T &__append_nonatomic() {
            auto *chunk = (Chunk *)dynamic_allocate(sizeof(Chunk));
            new (&chunk->m_data) T();
            auto *old_head = m_head;
            atomic_store(&m_head, chunk);
            chunk->m_next = old_head;
            return chunk->m_data;
        }

        template <class Func>
        inline FDB_DEVICE void foreach(Func func) const {
            for (auto chunk = m_head; chunk; chunk = chunk->m_next) {
                func(chunk->m_data);
            }
        }
    };

    HashMap<K, Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity() const {
        return m_grid.capacity();
    }

    inline void reserve(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear() {
        m_grid.clear();
    }

    struct View {
        typename HashMap<K, Leaf>::View m_view;

        View(HashListMap const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach_leaf(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach(kernel, cfg);
        }

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            parallel_foreach_leaf([=] FDB_DEVICE (K key, Leaf &leaf) {
                leaf.foreach([&] (T &value) {
                    kernel(std::as_const(key), value);
                });
            }, cfg);
        }

        inline FDB_DEVICE T &append(K key) const {
            auto *leaf = m_view.touch(key);
            return leaf->append();
        }

        inline FDB_DEVICE Leaf *probe_leaf(K key) const {
            return m_view.find(key);
        }

        inline FDB_DEVICE Leaf &touch_leaf(K key) const {
            return *m_view.touch(key);
        }

        inline FDB_DEVICE Leaf &get_leaf(K key) const {
            return *m_view.find(key);
        }
    };

    View view() const {
        return *this;
    }
};

}
