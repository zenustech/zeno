#pragma once

#include "HashGrid.h"

namespace fdb {

template <class T, size_t X>
struct H21Dx_Grid {
    struct Leaf {
        T m_data[1 << (X * 3)]{};

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] << X | coord[2] << (X * 2);
            return m_data[i];
        }
    };

    HashGrid<Leaf> m_grid;

    inline FDB_CONSTEXPR size_t capacity_leafs() const {
        return m_grid.capacity();
    }

    inline void reserve_leafs(size_t n) {
        m_grid.reserve(n);
    }

    inline void clear_leafs() {
        m_grid.clear();
    }

    struct View {
        typename HashGrid<Leaf>::View m_view;

        View(H21Dx_Grid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i leaf_coord, Leaf &leaf) {
                leaf_coord <<= X;
                for (int i = 0; i < (1 << X * 3); i++) {
                    int x = i & (1 << X) - 1;
                    int y = (i >> X) & (1 << X) - 1;
                    int z = (i >> (X * 2)) & (1 << X) - 1;
                    vec3i coord = leaf_coord | vec3i(x, y, z);
                    kernel(std::as_const(coord), leaf.m_data[i]);
                }
            }, cfg);
        }

        inline FDB_DEVICE T *probe(vec3i coord) const {
            auto *leaf = m_view.find(coord >> X);
            if (!leaf)
                return nullptr;
            return &leaf->at(coord & (1 << X) - 1);
        }

        inline FDB_DEVICE T &operator[](vec3i coord) const {
            auto *leaf = m_view.touch(coord >> X);
            return leaf->at(coord & (1 << X) - 1);
        }

        inline FDB_DEVICE T &operator()(vec3i coord) const {
            auto *leaf = m_view.find(coord >> X);
            return leaf->at(coord & (1 << X) - 1);
        }
    };

    View view() const {
        return *this;
    }
};

}
