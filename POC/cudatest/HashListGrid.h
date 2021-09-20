#pragma once

#include "HashMap_vec3i.h"
#include "HashListMap.h"

namespace fdb {

template <class T, size_t TileSize = 1024>
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

    struct Cell {
        typename HashListMap<vec3i, Tile>::Leaf m_leaf;

        inline FDB_DEVICE T &__append_slow() {
            T *ptr;
            atomic_spin_lock(&m_leaf.m_lock);
            if (!m_leaf.m_head || m_leaf.m_head->m_data.m_count >= TileSize) {
                Tile *tile = &m_leaf.__append_nonatomic();
                ptr = &tile->m_data[0];
                tile->m_count = 1;
            } else {
                Tile *tile = &m_leaf.m_head->m_data;
                int idx = tile->m_count++;
                ptr = &tile->m_data[idx];
            }
            atomic_spin_unlock(&m_leaf.m_lock);
            return *ptr;
        }

        inline FDB_DEVICE T &append() {
            if (auto *chunk = m_leaf.m_head; chunk) {
                Tile *tile = &chunk->m_data;
                int idx = atomic_add(&tile->m_count, 1);
                if (idx < TileSize) {
                    return tile->m_data[idx];
                } else {
                    atomic_store(&tile->m_count, (int)TileSize);
                }
            }
            return __append_slow();
        }

        template <class Func>
        inline FDB_DEVICE void foreach(Func func) const {
            m_leaf.foreach([&] (Tile &tile) {
                for (int i = 0; i < std::min((int)TileSize, tile.m_count); i++) {
                    func(tile.m_data[i]);
                }
            });
        }
    };

    struct View {
        typename HashListMap<vec3i, Tile>::View m_view;

        View(HashListGrid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            parallel_foreach_cell([=] FDB_DEVICE (vec3i coord, Cell &cell) {
                cell.foreach([&] (T &val) {
                    kernel(std::as_const(coord), val);
                });
            }, cfg);
        }

        template <class Kernel>
        inline void parallel_foreach_cell(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach_leaf([=] FDB_DEVICE (vec3i coord, auto &leaf) {
                Cell &cell = *(Cell *)&leaf;
                kernel(std::as_const(coord), cell);
            }, cfg);
        }

        inline FDB_DEVICE T &append(vec3i coord) const {
            auto *cell = &touch_cell(coord);
            return cell->append();
        }

        template <class Func>
        inline FDB_DEVICE void foreach_in_cell(vec3i coord, Func func) const {
            auto *cell = probe_cell(coord);
            if (cell) {
                cell->foreach(func);
            }
        }

        inline FDB_DEVICE Cell *probe_cell(vec3i coord) const {
            return (Cell *)m_view.probe_leaf(coord);
        }

        inline FDB_DEVICE Cell &touch_cell(vec3i coord) const {
            return *(Cell *)&m_view.touch_leaf(coord);
        }

        inline FDB_DEVICE Cell &get_cell(vec3i coord) const {
            return *(Cell *)&m_view.get_leaf(coord);
        }
    };

    View view() const {
        return *this;
    }
};

}
