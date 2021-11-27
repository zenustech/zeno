#pragma once

#include "HashListMap.h"
#include "Array.h"

namespace fdb {

template <class T, size_t TileSize = 128>
struct HashListGrid {
    static_assert(std::is_constructible<T>::value);
    static_assert(std::is_trivially_copy_constructible<T>::value);
    static_assert(std::is_trivially_copy_assignable<T>::value);
    static_assert(std::is_trivially_move_constructible<T>::value);
    static_assert(std::is_trivially_move_assignable<T>::value);
    static_assert(std::is_trivially_destructible<T>::value);

    struct Tile {
        Array<T, TileSize> m_data{};
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

    struct Leaf {
        typename HashListMap<vec3i, Tile>::Leaf m_leaf;

        inline FDB_DEVICE void append(T val) {
            /*if (auto *chunk = atomic_load(&m_leaf.m_head); chunk) {
                Tile *tile = &chunk->m_data;
                int idx = atomic_add(&tile->m_count, 1);
                printf("%d %d\n", idx, tile->m_count);
                if (idx < TileSize - 2) {
                    tile->m_data.store(idx, val);
                    return;
                }
                atomic_store(&tile->m_count, (int)TileSize);
            }*/
            atomic_spin_lock(&m_leaf.m_lock);
            if (!m_leaf.m_head || m_leaf.m_head->m_data.m_count >= TileSize) {
                Tile *tile = &m_leaf.__append_nonatomic();
                tile->m_data.store(0, val);
                tile->m_count = 1;
            } else {
                Tile *tile = &m_leaf.m_head->m_data;
                int idx = tile->m_count++;
                tile->m_data.store(idx, val);
            }
            atomic_spin_unlock(&m_leaf.m_lock);
        }

        template <class Func>
        inline FDB_DEVICE void foreach_load(Func func) const {
            m_leaf.foreach([&] (Tile &tile) {
                for (int i = 0; i < std::min((int)TileSize, tile.m_count); i++) {
                    func(tile.m_data.load(i));
                }
            });
        }

        template <class Func>
        inline FDB_DEVICE void foreach_store(Func func) const {
            m_leaf.foreach([&] (Tile &tile) {
                for (int i = 0; i < std::min((int)TileSize, tile.m_count); i++) {
                    tile.m_data.store(i, func());
                }
            });
        }

        template <class Func>
        inline FDB_DEVICE void foreach_modify(Func func) const {
            m_leaf.foreach([&] (Tile &tile) {
                for (int i = 0; i < std::min((int)TileSize, tile.m_count); i++) {
                    auto val = tile.m_data.load(i);
                    func(val);
                    tile.m_data.store(i, val);
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
            parallel_foreach_leaf([=] FDB_DEVICE (vec3i coord, Leaf &leaf) {
                leaf.foreach_modify([&] (T &val) {
                    kernel(std::as_const(coord), val);
                });
            }, cfg);
        }

        template <class Kernel>
        inline void parallel_foreach_leaf(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach_leaf([=] FDB_DEVICE (vec3i coord, auto &leaf) {
                kernel(std::as_const(coord), *(Leaf *)&leaf);
            }, cfg);
        }

        inline FDB_DEVICE void append(vec3i coord, T val) const {
            auto *leaf = &touch_leaf(coord);
            return leaf->append(val);
        }

        template <class Func>
        inline FDB_DEVICE void foreach_in_leaf(vec3i coord, Func func) const {
            auto *leaf = probe_leaf(coord);
            if (leaf) {
                leaf->foreach(func);
            }
        }

        inline FDB_DEVICE Leaf *probe_leaf(vec3i coord) const {
            return (Leaf *)m_view.probe_leaf(coord);
        }

        inline FDB_DEVICE Leaf &touch_leaf(vec3i coord) const {
            return *(Leaf *)&m_view.touch_leaf(coord);
        }

        inline FDB_DEVICE Leaf &get_leaf(vec3i coord) const {
            return *(Leaf *)&m_view.get_leaf(coord);
        }
    };

    View view() const {
        return *this;
    }
};

}
