#pragma once

#include "HashGrid.h"

namespace fdb {

template <class T>
struct HashListGrid {
    struct Chunk {
        T m_data{};
        Chunk *m_next{nullptr};
    };

    struct Leaf {
        Chunk *m_head{nullptr};

        T *insert(T val) {
            auto *new_node = (Chunk *)malloc(sizeof(Chunk));
            new (new_node) Chunk();
            auto *old_head = atomic_swap(&m_head, new_node);
            new_node->m_next = old_head;
            return &new_node->m_data;
        }
    };

    HashGrid<Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity_blocks() const {
        return m_grid.capacity();
    }

    inline void reserve_blocks(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear_blocks() {
        m_grid.clear_blocks();
    }

    struct View {
        typename HashGrid<Leaf>::View m_view;

        View(HashListGrid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i coord, Leaf &leaf) {
                for (; chain; chain = chain->next) {
                    kernel(std::as_const(coord), leaf->m_head->m_data);
                }
            }, cfg);
        }

        inline FDB_DEVICE T *insert(vec3i coord, T val) const {
            auto *leaf = m_view.find(coord);
            return leaf->insert(val);
        }

        inline FDB_DEVICE Leaf *probe_leaf_at(vec3i coord) const {
            return m_view.find(coord);
        }

        inline FDB_DEVICE Leaf &touch_leaf_at(vec3i coord) const {
            return *m_view.touch(coord);
        }

        inline FDB_DEVICE Leaf &leaf_at(vec3i coord) const {
            return *m_view.find(coord);
        }
    };

    View view() const {
        return *this;
    }
};

}
