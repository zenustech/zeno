#pragma once

#include "Coord.h"

namespace fdb {

template <class D, int L>
struct LeafBase {
    static Coord indexToCoord(int i) {
        int x = i & (1 << L) - 1;
        i >>= L;
        int y = i & (1 << L) - 1;
        i >>= L;
        int z = i & (1 << L) - 1;
        return {x, y, z};
    }

    static int coordToIndex(Coord const &coord) {
        int x = coord[0] & (1 << L) - 1;
        int y = coord[1] & (1 << L) - 1;
        int z = coord[2] & (1 << L) - 1;
        return z << L * 2 | y << L | x;
    }

    static int getElementCount() {
        return 1 << L;
    }

    template <class F>
    void cforeachElement(F const &f) const {
        auto *that = static_cast<D *>(this);
        for (int i = 0; i < that->getElementCount(); i++) {
            auto value = that->getValueAt(i);
            f(value, i);
        }
    }

    template <class F>
    void foreachElement(F const &f) {
        auto *that = static_cast<D *>(this);
        for (int i = 0; i < that->getElementCount(); i++) {
            auto value = that->getValueAt(i);
            f(value, i);
            that->setValueAt(i, value);
        }
    }
};

template <class T, int L>
struct Leaf : LeafBase<Leaf<T, L>, L> {
    using ValueType = T;

    T m_data[1 << L * 3];

    ValueType getValueAt(int i) const {
        return m_data[i];
    }

    void setValueAt(int i, ValueType const &value) {
        m_data[i] = value;
    }
};

}
