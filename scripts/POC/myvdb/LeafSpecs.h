#pragma once

#include "vec.h"
#include "Leaf.h"

template <class T, int N>
struct SOA {
};

template <class T, int N>
struct AOS {
};

template <class T, int N>
struct Points {
};

template <class T, int N, int L>
struct LeafNode<Points<T, N>, L>
    : LeafNodeBase<LeafNode<Points<T, N>, L>, L> {
    using ValueType = T;

    int m_pos[N];
    T m_data[N];
    int m_count = 0;
    LeafNode *m_next = nullptr;

    int getElementCount() const {
        return m_count;
    }

    LeafNode *insertElement(Coord const &coord,
            ValueType const &value) {
        if (m_count >= 1 << L) {
            if (!m_next)
                m_next = new LeafNode;
            return m_next->insertElement(coord, value);
        }
        int i = m_count++;
        m_pos[i] = coord.z << L * 2 | coord.y << L | coord.x;
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

template <class T, int N, int L>
struct LeafNode<AOS<T, N>, L>
    : LeafNodeBase<LeafNode<AOS<T, N>, L>, L> {
    using ValueType = std::array<T, N>;

    T m_data[1 << L * 3][N];

    ValueType getValueAt(int i) const {
        ValueType value;
        for (int d = 0; d < N; d++) {
            value[d] = m_data[d][i];
        }
        return value;
    }

    void setValueAt(int i, ValueType const &value) {
        for (int d = 0; d < N; d++) {
            m_data[d][i] = value[d];
        }
    }
};

template <class T, int N, int L>
struct LeafNode<SOA<T, N>, L>
    : LeafNodeBase<LeafNode<SOA<T, N>, L>, L> {
    using ValueType = std::array<T, N>;

    T m_data[N][1 << L * 3];

    ValueType getValueAt(int i) const {
        ValueType value;
        for (int d = 0; d < N; d++) {
            value[d] = m_data[i][d];
        }
        return value;
    }

    void setValueAt(int i, ValueType const &value) {
        for (int d = 0; d < N; d++) {
            m_data[i][d] = value[d];
        }
    }
};
