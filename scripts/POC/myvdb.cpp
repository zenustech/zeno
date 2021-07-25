#include <vector>

struct Coord {
    int x, y, z;
};

template <class T, int L3>
struct LeafNode {
    T m_data[1 << L3 * 3];
};

template <class T, int L2, int L3>
struct InternalNode {
    LeafNode<T, L3> *m_data[1 << L2 * 3];
};

template <class T, int L1, int L2, int L3>
struct RootNode {
    InternalNode<T, L2, L3> *m_data[1 << L1 * 3];
};

template <class T, int L1 = 5, int L2 = 4, int L3 = 3>
struct Tree {
    using ElementType = T;
    using RootNodeType = RootNode<T, L1, L2, L3>;
    using InternalNodeType = InternalNode<T, L2, L3>;
    using LeafNodeType = LeafNode<T, L3>;

    RootNodeType m_root;

    static int _linearizeRootToInternal(Coord const &coord) {
        int x = coord.x >> (L2 + L3) & (1 << L1) - 1;
        int y = coord.y >> (L2 + L3) & (1 << L1) - 1;
        int z = coord.z >> (L2 + L3) & (1 << L1) - 1;
        return z << L3 * 2 | y << L3 | x;
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

    static int _linearizeInternalToLeaf(Coord const &coord) {
        int x = coord.x >> L3 & (1 << L2) - 1;
        int y = coord.y >> L3 & (1 << L2) - 1;
        int z = coord.z >> L3 & (1 << L2) - 1;
        return z << L2 * 2 | y << L2 | x;
    }

    LeafNodeType *&leafAt(Coord const &coord) const {
        int i = _linearizeRootToInternal(coord);
        int j = _linearizeInternalToLeaf(coord);
        return m_root.m_data[i]->m_data[j];
    }

    template <class F>
    void foreachLeaf(F const &f) const {
        for (int i = 0; i < 1 << (L3 * 3); i++) {
            InternalNodeType &intern = m_root.m_data[i];
            for (int j = 0; j < 1 << (L2 * 3); j++) {
                LeafNodeType *&leaf = intern.m_data[j];
                Coord coord = _delinearizeToCoord(i, j);
                f(leaf, coord);
            }
        }
    }

    std::vector<Coord> getLeafs() const {
        std::vector<Coord> coords;
        foreachLeaf([&] (LeafNodeType *, Coord const &coord) {
            coords.push_back(coord);
        });
        return coords;
    }
};

int main() {
    //Tree<float> tree;
}
