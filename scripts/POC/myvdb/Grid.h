#pragma once

#include "Coord.h"

namespace fdb {

template <class T>
struct Grid {
    using GridType = T;
    using LeafType = typename T::LeafType;
    static constexpr auto LeafShift = T::LeafShift;

    float leaf_size = 1.f;
    GridType *m_tree = nullptr;

    Grid() {
        m_tree = new GridType;
    }

    Grid(Grid const &) = delete;

    ~Grid() {
        delete m_tree;
        m_tree = nullptr;
    }

    LeafType *leafAt(Coord const &coord) {
        return m_tree->leafAt(coord);
    }

    LeafType *cleafAt(Coord const &coord) const {
        return m_tree->cleafAt(coord);
    }

    template <class F>
    void foreachLeaf(F const &f) const {
        m_tree->foreachLeaf(f);
    }
};

}
