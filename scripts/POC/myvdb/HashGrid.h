#pragma once

#include "Leaf.h"

namespace fdb {

template <class T, int HL = 16, int L = 10>
struct HashGrid {
    static constexpr auto HashShift = HL;
    static constexpr auto LeafShift = L;
    using ElementType = T;
    using LeafType = Leaf<T, L>;

    struct HashEntry {
        Coord m_coord;
        HashEntry *m_next;
        LeafType *m_leaf;
    };

    HashEntry *m_entries[1 << HL];

    HashGrid() {
        for (int i = 0; i < 1 << HL; i++) {
            m_entries[i] = nullptr;
        }
    }

    static int _hashCoord(Coord const &coord) {
        int x = coord[0];
        int y = coord[1];
        int z = coord[2];
        int h = (73856093 * x) ^ (19349663 * y) ^ (83492791 * z);
        return h & (1 << HL) - 1;
    }

    LeafType *leafAt(Coord const &coord) {
        int i = _hashCoord(coord);
        for (auto *curr = m_entries[i]; curr; curr = curr->m_next) {
            if (all(curr->m_coord == coord)) {
                return curr->m_leaf;
            }
        }
        auto *entry = new HashEntry;
        auto *leaf = new LeafType;
        entry->m_coord = coord;
        entry->m_next = m_entries[i];
        entry->m_leaf = leaf;
        m_entries[i] = entry;
        return leaf;
    }

    LeafType *cleafAt(Coord const &coord) const {
        int i = _hashCoord(coord);
        for (auto *curr = m_entries[i]; curr; curr = curr->m_next) {
            if (curr->m_coord == coord) {
                return curr->m_leaf;
            }
        }
        return nullptr;
    }

    template <class F>
    void foreachLeaf(F const &f) const {
        for (int i = 0; i < 1 << HL; i++) {
            for (auto *curr = m_entries[i]; curr; curr = curr->m_next) {
                f(curr->m_leaf, curr->m_coord);
            }
        }
    }
};

}
