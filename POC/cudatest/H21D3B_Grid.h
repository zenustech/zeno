#pragma once

#include "HashGrid.h"

namespace fdb {

template <class T>
struct H21D3B_Grid {
    struct Leaf {
        T m_data[8 * 8 * 8]{};
        uint8_t m_mask[8 * 8]{};

        inline FDB_DEVICE bool is_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            return m_mask[i] & (1 << coord[0]);
        }

        inline FDB_DEVICE void turn_on(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
        }

        inline FDB_DEVICE void turn_off(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] &= ~(1 << coord[0]);
        }

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] | coord[2] << 6;
            return m_data[i];
        }

        inline FDB_DEVICE T &turn_on_at(vec3i coord) {
            size_t i = coord[1] | coord[2] << 3;
            m_mask[i] |= 1 << coord[0];
            return m_data[i << 3 | coord[0]];
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

        View(H21D3B_Grid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                leaf_coord <<= 3;
                for (int i = 0; i < 8 * 8 * 8; i++) {
                    int x = i & 0x7;
                    if (leaf.m_mask[i >> 3] & (1 << x)) {
                        int y = (i >> 3) & 0x7;
                        int z = (i >> 6) & 0x7;
                        vec3i coord = leaf_coord | vec3i(x, y, z);
                        kernel(std::as_const(coord), leaf.m_data[i]);
                    }
                }
            }, cfg);
        }

        template <bool allow_off = false>
        inline FDB_DEVICE T *probe(vec3i coord) const {
            auto *leaf = m_view.find(coord >> 3);
            if (!leaf)
                return nullptr;
            coord &= 0x7;
            if (!allow_off && !leaf->is_on(coord))
                return nullptr;
            return &leaf->at(coord);
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            auto *leaf = m_view.touch(coord >> 3);
            return leaf->turn_on_at(coord & 0x7);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            auto *leaf = m_view.find(coord >> 3);
            return leaf->at(coord & 0x7);
        }
    };

    View view() const {
        return *this;
    }
};

}
