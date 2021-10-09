#pragma once

#include "H21PyDx_Grid.h"

namespace fdb {

template <class T, size_t X, size_t Y>
struct H21PyDx_Grid {
    struct Leaf {
        T m_data[1 << (X * 3)]{};

        inline FDB_DEVICE T &at(vec3i coord) {
            size_t i = coord[0] | coord[1] << X | coord[2] << (X * 2);
            return m_data[i];
        }
    };

    struct Block {
        Leaf *m_ptrs[1 << (Y * 3)]{};

        inline FDB_DEVICE Leaf *&at(vec3i coord) {
            size_t i = coord[0] | coord[1] << Y | coord[2] << (Y * 2);
            return m_ptrs[i];
        }
    };

    HashGrid<Block> m_grid;

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
        typename H21Dx_Grid<Leaf>::View m_view;

        View(H21PyDx_Grid const &parent)
            : m_view(parent.m_grid.view())
        {}

        template <class Kernel>
        inline void parallel_foreach(Kernel kernel, ParallelConfig cfg = {256, 2}) const {
            ConcurrentVector<vec3i> list;
            auto listv = list.view();
            m_view.parallel_foreach([=] FDB_DEVICE (vec3i block_coord, Block &block) {
                block_coord <<= Y;
                for (int i = 0; i < (1 << Y * 3); i++) {
                    auto leaf = block.m_ptrs[i];
                    if (leaf) {
                        int x = i & (1 << Y) - 1;
                        int y = (i >> Y) & (1 << Y) - 1;
                        int z = (i >> (Y * 2)) & (1 << Y) - 1;
                        vec3i leaf_coord = block_coord | vec3i(x, y, z);
                        listv.push_back(leaf_coord);
                    }
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
