#pragma once

#include "Leaf.h"

namespace fdb {

template <class T, int N>
struct Points {
};

template <class T, int N, int L>
struct Leaf<Points<T, N>, L> : LeafBase<Leaf<Points<T, N>, L>, L> {
    using ValueType = T;

    int m_pos[N];
    T m_data[N];
    int m_count = 0;
    Leaf *m_next = nullptr;

    int getElementCount() const {
        return m_count;
    }

    Leaf *insertElement(Coord const &coord,
            ValueType const &value) {
        if (m_count >= 1 << L) {
            if (!m_next)
                m_next = new Leaf;
            return m_next->insertElement(coord, value);
        }
        int i = m_count++;
        m_pos[i] = coord[2] << L * 2 | coord[1] << L | coord[0];
        m_data[i] = value;
        return this;
    }

    Coord indexToCoord(int i) const {
        auto pos = m_pos[i];
        int x = pos & (1 << L) - 1;
        pos >>= L;
        int y = pos & (1 << L) - 1;
        pos >>= L;
        int z = pos & (1 << L) - 1;
        return {x, y, z};
    }

    static void coordToIndex(Coord const &) {}

    ValueType getValueAt(int i) const {
        return m_data[i];
    }

    void setValueAt(int i, ValueType const &value) {
        m_data[i] = value;
    }
};

}
