#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>
#include <array>

struct Coord {
    int x, y, z;
};

template <int L>
static Coord combineCoord(Coord const &leafCoord, Coord const &subCoord) {
    return {
        leafCoord.x << L | subCoord.x,
        leafCoord.y << L | subCoord.y,
        leafCoord.z << L | subCoord.z,
    };
}

template <int L>
static Coord staggerCoord(Coord const &coord) {
    int offset = 1 << (L - 1);
    return {
        coord.x + offset,
        coord.y + offset,
        coord.z + offset,
    };
}

template <class D, int L>
struct LeafNodeBase {
    static Coord indexToCoord(int i) {
        int x = i & (1 << L) - 1;
        i >>= L;
        int y = i & (1 << L) - 1;
        i >>= L;
        int z = i & (1 << L) - 1;
        return {x, y, z};
    }

    static int coordToIndex(Coord const &coord) {
        int x = coord.x & (1 << L) - 1;
        int y = coord.y & (1 << L) - 1;
        int z = coord.z & (1 << L) - 1;
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
struct LeafNode
    : LeafNodeBase<LeafNode<T, L>, L> {
    using ValueType = T;

    T m_data[1 << L * 3];

    ValueType getValueAt(int i) const {
        return m_data[i];
    }

    void setValueAt(int i, ValueType const &value) {
        m_data[i] = value;
    }
};

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

template <class T, int L2, int L>
struct InternalNode {
    LeafNode<T, L> *m_data[1 << L2 * 3];

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
    using LeafNodeType = LeafNode<T, L>;

    RootNodeType *m_root;

    P2PGrid() {
        m_root = new RootNodeType;
    }

    ~P2PGrid() {
        delete m_root;
        m_root = nullptr;
    }

    static int _linearizeRootToInternal(Coord const &coord) {
        int x = coord.x >> L2 & (1 << L1) - 1;
        int y = coord.y >> L2 & (1 << L1) - 1;
        int z = coord.z >> L2 & (1 << L1) - 1;
        return z << L * 2 | y << L | x;
    }

    static int _linearizeInternalToLeaf(Coord const &coord) {
        int x = coord.x & (1 << L2) - 1;
        int y = coord.y & (1 << L2) - 1;
        int z = coord.z & (1 << L2) - 1;
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

    LeafNodeType *cleafAt(Coord const &coord) const {
        int internalIdx = _linearizeRootToInternal(coord);
        int leafIdx = _linearizeInternalToLeaf(coord);
        auto *internalNode = m_root->m_data[internalIdx];
        if (!internalNode) {
            return nullptr;
        }
        auto *leafNode = internalNode->m_data[leafIdx];
        return leafNode;
    }

    LeafNodeType *leafAt(Coord const &coord) const {
        int internalIdx = _linearizeRootToInternal(coord);
        int leafIdx = _linearizeInternalToLeaf(coord);
        auto *&internalNode = m_root->m_data[internalIdx];
        if (!internalNode) {
            internalNode = new InternalNodeType;
        }
        auto *&leafNode = internalNode->m_data[leafIdx];
        if (!leafNode) {
            leafNode = new LeafNodeType;
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

template <class T, int HL = 16, int L = 10>
struct HashGrid {
    static constexpr auto HashShift = HL;
    static constexpr auto LeafShift = L;
    using ElementType = T;
    using LeafNodeType = LeafNode<T, L>;

    struct HashEntry {
        Coord m_coord;
        HashEntry *m_next;
        LeafNodeType *m_leaf;
    };

    HashEntry *m_entries[1 << HL];

    HashGrid() {
        for (int i = 0; i < 1 << HL; i++) {
            m_entries[i] = nullptr;
        }
    }

    static int _hashCoord(Coord const &coord) {
        int x = coord.x;
        int y = coord.y;
        int z = coord.z;
        int h = (73856093 * x) ^ (19349663 * y) ^ (83492791 * z);
        return h & (1 << HL) - 1;
    }

    LeafNodeType *leafAt(Coord const &coord) {
        int i = _hashCoord(coord);
        for (auto *curr = m_entries[i]; curr; curr = curr->m_next) {
            if (curr->m_coord.x == coord.x
             && curr->m_coord.y == coord.y
             && curr->m_coord.z == coord.z) {
                return curr->m_leaf;
            }
        }
        auto *entry = new HashEntry;
        auto *leaf = new LeafNodeType;
        entry->m_coord = coord;
        entry->m_next = m_entries[i];
        entry->m_leaf = leaf;
        m_entries[i] = entry;
        return leaf;
    }

    LeafNodeType *cleafAt(Coord const &coord) const {
        int i = _hashCoord(coord);
        for (auto *curr = m_entries[i]; curr; curr = curr->m_next) {
            if (curr->m_coord.x == coord.x
             && curr->m_coord.y == coord.y
             && curr->m_coord.z == coord.z) {
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

int main() {
    HashGrid<Points<int, 32>> grid, new_grid;

    const int L = grid.LeafShift;
    const int N = 1;
    std::vector<glm::vec3> pos(N);
    std::vector<glm::vec3> new_pos(N);
    std::vector<glm::vec3> vel(N);
    for (int i = 0; i < N; i++) {
        pos[i] = (glm::vec3)glm::ivec3(
                rand(), rand(), rand()) / (float)RAND_MAX;
        vel[i] = (glm::vec3)glm::ivec3(
                rand(), rand(), rand()) / (float)RAND_MAX * 2.f - 1.f;
    }

    float leaf_size = 0.04f;
    float dt = 0.01f;

    // p2g
    for (int i = 0; i < N; i++) {
        auto iipos = pos[i] / leaf_size;
        auto ipos = glm::ivec3(iipos);
        auto jpos = glm::ivec3(glm::mod(iipos, 1.f) * float(1 << L));
        Coord leafCoord{ipos.x, ipos.y, ipos.z};
        auto *leaf = grid.leafAt(leafCoord);
        Coord subCoord{jpos.x, jpos.y, jpos.z};
        leaf->insertElement(subCoord, i);
        printf("? %d %d %d\n", leafCoord.x, leafCoord.y, leafCoord.z);
        printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
    }

    // advect
    grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            auto vel_dt = vel[value] * dt;
            subCoord.x += int(vel_dt.x * float(1 << L) / leaf_size);
            subCoord.y += int(vel_dt.y * float(1 << L) / leaf_size);
            subCoord.z += int(vel_dt.z * float(1 << L) / leaf_size);

            Coord newLeafCoord = leafCoord;
            if (subCoord.x < 0) newLeafCoord.x--; else if (subCoord.x >= 1 << L) newLeafCoord.x++;
            if (subCoord.y < 0) newLeafCoord.y--; else if (subCoord.y >= 1 << L) newLeafCoord.y++;
            if (subCoord.z < 0) newLeafCoord.z--; else if (subCoord.z >= 1 << L) newLeafCoord.z++;

            subCoord.x &= (1 << L) - 1;
            subCoord.y &= (1 << L) - 1;
            subCoord.z &= (1 << L) - 1;

            printf("? %d %d %d\n", newLeafCoord.x, newLeafCoord.y, newLeafCoord.z);
            printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
            new_grid.leafAt(newLeafCoord)->insertElement(subCoord, value);
        });
    });

    // g2p
    new_grid.foreachLeaf([&] (auto *leaf, Coord const &leafCoord) {
        printf("? %d %d %d\n", leafCoord.x, leafCoord.y, leafCoord.z);
        leaf->foreachElement([&] (auto &value, int index) {
            Coord subCoord = leaf->indexToCoord(index);
            printf("! %d %d %d\n", subCoord.x, subCoord.y, subCoord.z);
            float fx = (leafCoord.x + subCoord.x / float(1 << L)) * leaf_size;
            float fy = (leafCoord.y + subCoord.y / float(1 << L)) * leaf_size;
            float fz = (leafCoord.z + subCoord.z / float(1 << L)) * leaf_size;

            new_pos[value] = glm::vec3(fx, fy, fz);
        });
    });

    for (int i = 0; i < N; i++) {
        printf("%f %f\n", new_pos[i].x - pos[i].x, vel[i].x * dt);
        printf("%f %f\n", new_pos[i].y - pos[i].y, vel[i].y * dt);
        printf("%f %f\n", new_pos[i].z - pos[i].z, vel[i].z * dt);
    }
}
//for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
