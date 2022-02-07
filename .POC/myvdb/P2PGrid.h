#pragma once

namespace fdb {

template <class T, int L2, int L>
struct InternalNode {
    Leaf<T, L> *m_data[1 << L2 * 3];

    InternalNode() {
        for (int i = 0; i < 1 << L2 * 3; i++) {
            m_data[i] = nullptr;
        }
    }

    ~InternalNode() {
        for (int i = 0; i < 1 << L2 * 3; i++) {
            if (m_data[i]) {
                delete m_data[i];
                m_data[i] = nullptr;
            }
        }
    }
};

template <class T, int L1, int L2, int L>
struct RootNode {
    InternalNode<T, L2, L> *m_data[1 << L1 * 3];

    RootNode() {
        for (int i = 0; i < 1 << L1 * 3; i++) {
            m_data[i] = nullptr;
        }
    }

    ~RootNode() {
        for (int i = 0; i < 1 << L1 * 3; i++) {
            if (m_data[i]) {
                delete m_data[i];
                m_data[i] = nullptr;
            }
        }
    }
};

template <class T, int L1 = 5, int L2 = 4, int L = 3>
struct P2PGrid {
    static constexpr auto RootShift = L1;
    static constexpr auto InternalShift = L2;
    static constexpr auto LeafShift = L;
    using ElementType = T;
    using RootNodeType = RootNode<T, L1, L2, L>;
    using InternalNodeType = InternalNode<T, L2, L>;
    using LeafType = Leaf<T, L>;

    RootNodeType *m_root;

    P2PGrid() {
        m_root = new RootNodeType;
    }

    ~P2PGrid() {
        delete m_root;
        m_root = nullptr;
    }

    static int _linearizeRootToInternal(Coord const &coord) {
        int x = coord[0] >> L2 & (1 << L1) - 1;
        int y = coord[1] >> L2 & (1 << L1) - 1;
        int z = coord[2] >> L2 & (1 << L1) - 1;
        return z << L * 2 | y << L | x;
    }

    static int _linearizeInternalToLeaf(Coord const &coord) {
        int x = coord[0] & (1 << L2) - 1;
        int y = coord[1] & (1 << L2) - 1;
        int z = coord[2] & (1 << L2) - 1;
        return z << L2 * 2 | y << L2 | x;
    }

    static Coord _delinearizeToCoord(int i, int j) {
        int ix = i & (1 << L2) - 1;
        i >>= L2;
        int iy = i & (1 << L2) - 1;
        i >>= L2;
        int iz = i & (1 << L2) - 1;

        int jx = j & (1 << L2) - 1;
        j >>= L2;
        int jy = j & (1 << L2) - 1;
        j >>= L2;
        int jz = j & (1 << L2) - 1;

        int x = ix << (L2 + L1) | jx << L1;
        int y = iy << (L2 + L1) | jy << L1;
        int z = iz << (L2 + L1) | jz << L1;

        return {x, y, z};
    }

    LeafType *cleafAt(Coord const &coord) const {
        int internalIdx = _linearizeRootToInternal(coord);
        int leafIdx = _linearizeInternalToLeaf(coord);
        auto *internalNode = m_root->m_data[internalIdx];
        if (!internalNode) {
            return nullptr;
        }
        auto *leafNode = internalNode->m_data[leafIdx];
        return leafNode;
    }

    LeafType *leafAt(Coord const &coord) const {
        int internalIdx = _linearizeRootToInternal(coord);
        int leafIdx = _linearizeInternalToLeaf(coord);
        auto *&internalNode = m_root->m_data[internalIdx];
        if (!internalNode) {
            internalNode = new InternalNodeType;
        }
        auto *&leafNode = internalNode->m_data[leafIdx];
        if (!leafNode) {
            leafNode = new LeafType;
        }
        return leafNode;
    }

    template <class F>
    void foreachLeaf(F const &f) const {
        for (int i = 0; i < 1 << L * 3; i++) {
            auto *intern = m_root->m_data[i];
            if (!intern) continue;
            for (int j = 0; j < 1 << L2 * 3; j++) {
                auto *leaf = intern->m_data[j];
                if (!leaf) continue;
                Coord coord = _delinearizeToCoord(i, j);
                f(leaf, coord);
            }
        }
    }
};

}
