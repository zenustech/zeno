#pragma once

#include "vec.h"
#include "Leaf.h"

namespace fdb {

template <class T, int N>
struct SOA {
};

template <class T, int N, int L>
struct Leaf<SOA<T, N>, L> : LeafBase<Leaf<SOA<T, N>, L>, L> {
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

}
