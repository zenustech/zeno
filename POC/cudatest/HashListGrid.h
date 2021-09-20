#pragma once

#include "HashMap_vec3i.h"
#include "HashListMap.h"

namespace fdb {

template <class T, size_t TileSize = 64>
struct HashListGrid {
    struct Tile {
        T m_data[TileSize]{};
        int m_count{0};
    };

    HashListMap<vec3i, Tile> m_grid;

    inline FDB_CONSTEXPR size_t capacity_blocks() const {
        return m_grid.capacity();
    }

    inline void reserve_blocks(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear_blocks() {
        m_grid.clear();
    }

    struct View {
        typename HashListMap<vec3i, Tile>::View m_view;

        View(HashListGrid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i coord, Tile &tile) {
                for (int i = 0; i < std::min((int)TileSize, tile.m_count); i++) {
                    kernel(std::as_const(coord), tile.m_data[i]);
                }
            }, cfg);
        }

        inline FDB_DEVICE T *append(vec3i coord) const {
            auto *leaf = &m_view.touch_leaf_at(coord);
            /*if (auto *chunk = leaf->m_head; chunk) {
                auto *tile = &chunk->m_data;
                auto idx = atomic_add(&tile->m_count, 1);
                if (idx < TileSize) {
                    return &tile->m_data[idx];
                } else {
                    atomic_store(&tile->m_count, (int)TileSize);
                }
            }*/

            T *ptr;
            atomic_spin_lock(&leaf->m_lock);
            if (!leaf->m_head || leaf->m_head->m_data.m_count >= TileSize) {
                auto *tile = leaf->append_nonatomic();
                ptr = &tile->m_data[0];
                tile->m_count = 1;
            } else {
                auto *tile = &leaf->m_head->m_data;
                auto idx = tile->m_count++;
                ptr = &tile->m_data[idx];
            }
            atomic_spin_unlock(&leaf->m_lock);
            return ptr;
        }

        inline FDB_DEVICE auto *probe_leaf_at(vec3i coord) const {
            return m_view.probe_leaf_at(coord);
        }

        inline FDB_DEVICE auto &touch_leaf_at(vec3i coord) const {
            return m_view.touch_leaf_at(coord);
        }

        inline FDB_DEVICE auto &leaf_at(vec3i coord) const {
            return m_view.leaf_at(coord);
        }
    };

    View view() const {
        return *this;
    }
};

}
