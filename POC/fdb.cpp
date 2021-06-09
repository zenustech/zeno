#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <cassert>
#include <omp.h>

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)


template <class Leaf_>
struct BaseGrid {
    using Leaf = Leaf_;

    struct InternalNode {
        Leaf *m[16 * 16 * 16];

        InternalNode() {
            std::memset(m, 0, sizeof(m));
        }

        ~InternalNode() {
            for (long i = 0; i < 16 * 16 * 16; i++) {
                if (m[i]) {
                    delete m[i];
                    m[i] = nullptr;
                }
            }
        }
    };

    InternalNode *m[32 * 32 * 32];

    BaseGrid() {
        std::memset(m, 0, sizeof(m));
    }

    ~BaseGrid() {
        clear_leaves();
    }

    void clear_leaves() {
        for (long i = 0; i < 32 * 32 * 32; i++) {
            if (m[i]) {
                delete m[i];
                m[i] = nullptr;
            }
        }
    }

    Leaf *w_leaf_at(long i, long j, long k) {
        long ii = i / 16;
        long jj = j / 16;
        long kk = k / 16;
        long mm = ii + 32 * (jj + 32 * kk);
        InternalNode *&n = m[mm];
        if (!n) {
            n = new InternalNode();
        }
        ii = i % 16;
        jj = j % 16;
        kk = k % 16;
        mm = ii + 16 * (jj + kk * 16);
        Leaf *&p = n->m[mm];
        if (!p) {
            p = new Leaf();
        }
        return p;
    }

    Leaf *p_leaf_at(long i, long j, long k) const {
        long ii = i / 16;
        long jj = j / 16;
        long kk = k / 16;
        long mm = ii + 32 * (jj + 32 * kk);
        InternalNode *n = m[mm];
        if (!n) {
            return nullptr;
        }
        ii = i % 16;
        jj = j % 16;
        kk = k % 16;
        mm = ii + 16 * (jj + kk * 16);
        Leaf *p = n->m[mm];
        return p;
    }

    Leaf *leaf_at(long i, long j, long k) const {
        long ii = i / 16;
        long jj = j / 16;
        long kk = k / 16;
        long mm = ii + 32 * (jj + 32 * kk);
        InternalNode *n = m[mm];
        ii = i % 16;
        jj = j % 16;
        kk = k % 16;
        mm = ii + 16 * (jj + kk * 16);
        Leaf *p = n->m[mm];
        return p;
    }

    template <class Func>
    void foreach_leaf(Func const &callback) const {
        for (long i = 0; i < 32 * 32 * 32; i++) {
            InternalNode *n = m[i];
            if (!n) continue;
            long ii = i % 32;
            long ij = i / 32 % 32;
            long ik = i / 32 / 32;
            for (long j = 0; j < 16 * 16 * 16; j++) {
                Leaf *p = n->m[j];
                if (!p) continue;
                long ji = j % 16;
                long jj = j / 16 % 16;
                long jk = j / 16 / 16;
                callback(p, ii * 16 + ji, ij * 16 + jj, ik * 16 + jk);
            }
        }
    }

    auto get_leaves() const {
        std::vector<std::tuple<Leaf *, long, long, long>> leaves;
        this->foreach_leaf([&] (auto *p, long ii, long jj, long kk) {
            leaves.emplace_back(p, ii, jj, kk);
        });
        return leaves;
    }
};


template <class T_>
struct DataLeaf {
    using T = T_;

    T m[8 * 8 * 8];

    DataLeaf() {
        std::memset(m, 0, sizeof(T) * 8 * 8 * 8);
    }

    T &w_at(long i, long j, long k) {
        return m[linearize(i, j, k)];
    }

    T *p_at(long i, long j, long k) const {
        return static_cast<T *>(&at(i, j, k));
    }

    T const &at(long i, long j, long k) const {
        return m[linearize(i, j, k)];
    }

    long linearize(long i, long j, long k) const {
        return i + 8 * (j + 8 * k);
    }
};


template <class Leaf>
struct BitmaskedLeaf : Leaf {
    using T = typename Leaf::T;

    unsigned char mask[8 * 8];

    BitmaskedLeaf() {
        std::memset(mask, 0, sizeof(unsigned char) * 8 * 8);
    }

    T &w_at(long i, long j, long k) {
        activate(i, j, k);
        return Leaf::w_at(i, j, k);
    }

    T *p_at(long i, long j, long k) const {
        if (!is_active(i, j, k))
            return nullptr;
        return Leaf::p_at(i, j, k);
    }

    using Leaf::at;

    bool is_active(long i, long j, long k) const {
        return (mask[j + k * 8] & (1 << i)) != 0;
    }

    void activate(long i, long j, long k) {
        mask[j + k * 8] |= (1 << i);
    }

    void deactivate(long i, long j, long k) {
        mask[j + k * 8] &= ~(1 << i);
    }
};


template <class Leaf_>
struct VolumeGrid : BaseGrid<Leaf_> {
    using Leaf = Leaf_;
    using T = typename Leaf::T;

    using BaseGrid<Leaf_>::w_leaf_at;
    using BaseGrid<Leaf_>::p_leaf_at;
    using BaseGrid<Leaf_>::leaf_at;

    T &w_at(long i, long j, long k) {
        auto *leaf = w_leaf_at(i / 8, j / 8, k / 8);
        return leaf->w_at(i % 8, j % 8, k % 8);
    }

    T *p_at(long i, long j, long k) const {
        auto *leaf = p_leaf_at(i / 8, j / 8, k / 8);
        if (!leaf)
            return nullptr;
        return leaf->p_at(i % 8, j % 8, k % 8);
    }

    T const &at(long i, long j, long k) const {
        auto *leaf = leaf_at(i / 8, j / 8, k / 8);
        return leaf->at(i % 8, j % 8, k % 8);
    }
};


int main(void)
{
    VolumeGrid<DataLeaf<float>> f;

    for (long k = 256-16; k < 256+16; k++) {
        for (long j = 256-16; j < 256+16; j++) {
            for (long i = 256-16; i < 256+16; i++) {
                //cout << i << ',' << j << ',' << k << endl;
                (void)f.w_leaf_at(i, j, k);
            }
        }
    }

    f.w_leaf_at(256, 256, 256)->w_at(4, 4, 4) = 32;
    show(f.get_leaves().size());

    auto t0 = std::chrono::steady_clock::now();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;
    return 0;
}
