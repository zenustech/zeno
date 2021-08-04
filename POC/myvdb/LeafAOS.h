#pragma once

#include "vec.h"
#include "Leaf.h"

namespace fdb {

template <class T, int N>
struct AOS {
};

template <class T, int N, int L>
struct Leaf<AOS<T, N>, L> : LeafBase<Leaf<AOS<T, N>, L>, L> {
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

}
