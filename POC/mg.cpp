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


inline std::tuple<long, long, long> unlinearize(long N, long idx) {
    return std::make_tuple<long, long, long>(
        idx % N, idx / N % N, idx / N / N);
}


template <class Leaf>
struct VDBGrid {
    using LeafType = Leaf;

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

    VDBGrid() {
        std::memset(m, 0, sizeof(m));
    }

    ~VDBGrid() {
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

    Leaf *leaf_at(long i, long j, long k) {
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

    Leaf *ro_leaf_at(long i, long j, long k) const {
        if (i < 0 || j < 0 || k < 0)
            return nullptr;
        if (i >= 512 || j >= 512 || k >= 512)
            return nullptr;
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


template <class Leaf>
struct BitmaskedLeaf : Leaf {
    unsigned char mask[8 * 8];

    inline bool is_active(long i, long j, long k) {
        return (mask[j + k * 8] & (1 << i)) != 0;
    }

    inline void activate(long i, long j, long k) {
        mask[j +k * 8] |= (1 << i);
    }

    inline void deactivate(long i, long j, long k) {
        mask[j + k * 8] &= ~(1 << i);
    }
};


template <class T>
struct BoundaryLeaf {
    using ValueType = T;

    T m[10 * 10 * 10];

    BoundaryLeaf() {
        std::memset(m, 0, sizeof(m));
    }

    inline T &at(long i, long j, long k) {
        return m[linearize(i, j, k)];
    }

    inline T const &at(long i, long j, long k) const {
        return m[linearize(i, j, k)];
    }

    inline long linearize(long i, long j, long k) const {
        return i + 1 + 10 * (j + 1 + 10 * (k + 1));
    }

    T around_at(int i, int j, int k) const {
        T x1 = at(i - 1, j, k);
        T x2 = at(i + 1, j, k);
        T y1 = at(i, j - 1, k);
        T y2 = at(i, j + 1, k);
        T z1 = at(i, j, k - 1);
        T z2 = at(i, j, k + 1);
        T c = at(i, j, k);
        T val = 6 * c - (x1 + x2 + y1 + y2 + z1 + z2);
        return val;
    }
};


template <class T>
struct SimpleLeaf {
    using ValueType = T;

    T m[8 * 8 * 8];

    SimpleLeaf() {
        std::memset(m, 0, sizeof(T) * 8 * 8 * 8);
    }

    inline T &at(long i, long j, long k) {
        return m[linearize(i, j, k)];
    }

    inline T const &at(long i, long j, long k) const {
        return m[linearize(i, j, k)];
    }

    inline long linearize(long i, long j, long k) const {
        return i + 8 * (j + 8 * k);
    }
};


template <class T>
struct RBGrid : VDBGrid<BoundaryLeaf<T>> {
    static constexpr long N = 8;

    void sync_boundaries() const {
        #pragma omp parallel for
        for (auto [leaf, ii, jj, kk]: this->get_leaves()) {
            if (auto *other = this->ro_leaf_at(ii + 1, jj, kk); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = other->at(0, i, j);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = T(0);
            if (auto *other = this->ro_leaf_at(ii - 1, jj, kk); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(-1, i, j) = other->at(7, i, j);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(-1, i, j) = T(0);
            if (auto *other = this->ro_leaf_at(ii, jj + 1, kk); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = other->at(0, i, j);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = T(0);
            if (auto *other = this->ro_leaf_at(ii, jj - 1, kk); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(i, -1, j) = other->at(i, 7, j);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(i, -1, j) = T(0);
            if (auto *other = this->ro_leaf_at(ii, jj, kk + 1); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = other->at(0, i, j);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(8, i, j) = T(0);
            if (auto *other = this->ro_leaf_at(ii, jj, kk - 1); other)
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(i, j, -1) = other->at(i, j, 7);
            else
                for (long j = 0; j < 8; j++)
                    for (long i = 0; i < 8; i++)
                        leaf->at(i, j, -1) = T(0);
        }
    }

    void copy_topology(RBGrid const &src) {
        for (auto [src_leaf, ii, jj, kk]: src.get_leaves()) {
            (void)this->leaf_at(ii, jj, kk);
        }
    }

    long leaf_count() const {
        return this->get_leaves().size();
    }

    void smooth(RBGrid const &rhs, long times) {
        for (long t = 0; t < times; t++) {
            this->sync_boundaries();
            #pragma omp parallel for
            for (auto [leaf, ii, jj, kk]: this->get_leaves()) {
                //show(omp_get_thread_num());
                auto *rhs_leaf = rhs.ro_leaf_at(ii, jj, kk);
                assert(rhs_leaf);
                for (long k = 0; k < N; k++) {
                    for (long j = 0; j < N; j++) {
                        for (long i = (j + k + t) % 2; i < N; i += 2) {
                            T x1 = leaf->at(i - 1, j, k);
                            T x2 = leaf->at(i + 1, j, k);
                            T y1 = leaf->at(i, j - 1, k);
                            T y2 = leaf->at(i, j + 1, k);
                            T z1 = leaf->at(i, j, k - 1);
                            T z2 = leaf->at(i, j, k + 1);
                            T f = rhs_leaf->at(i, j, k);
                            T val = x1 + x2 + y1 + y2 + z1 + z2 + f;
                            leaf->at(i, j, k) = val / 6;
                        }
                    }
                }
            }
        }
    }

    T normsqr() const {
        T res(0);
        for (auto [leaf, ii, jj, kk]: this->get_leaves()) {
            for (long _ = 0; _ < N * N * N; _++) {
                auto [i, j, k] = unlinearize(N, _);
                T val = leaf->at(i, j, k);
                res += val * val;
            }
        }
        return res;
    }

    void residual(RBGrid &out, RBGrid const &rhs) const {
        this->sync_boundaries();
        #pragma omp parallel for
        for (auto [leaf, ii, jj, kk]: this->get_leaves()) {
            auto *out_leaf = out.leaf_at(ii, jj, kk);
            auto *rhs_leaf = rhs.ro_leaf_at(ii, jj, kk);
            assert(rhs_leaf);
            for (long _ = 0; _ < N * N * N; _++) {
                auto [i, j, k] = unlinearize(N, _);
                T val = rhs_leaf->at(i, j, k) - leaf->around_at(i, j, k);
                out_leaf->at(i, j, k) = val;
            }
        }
    }

    void add(RBGrid const &src) {
        #pragma omp parallel for
        for (auto [src_leaf, ii, jj, kk]: src.get_leaves()) {
            auto *leaf = this->leaf_at(ii, jj, kk);
            for (long k = 0; k < N; k++) {
                for (long j = 0; j < N; j++) {
                    for (long i = 0; i < N; i++) {
                        leaf->at(i, j, k) += src_leaf->at(i, j, k);
                    }
                }
            }
        }
    }

    void restrict(RBGrid const &src) {
        for (auto [src_leaf, ii, jj, kk]: src.get_leaves()) {
            auto *leaf = this->leaf_at(ii / 2, jj / 2, kk / 2);
            for (long k = 0; k < N / 2; k++) {
                for (long j = 0; j < N / 2; j++) {
                    for (long i = 0; i < N / 2; i++) {
                        T ooo = src_leaf->at(i * 2 + 0, j * 2 + 0, k * 2 + 0);
                        T ioo = src_leaf->at(i * 2 + 1, j * 2 + 0, k * 2 + 0);
                        T iio = src_leaf->at(i * 2 + 1, j * 2 + 1, k * 2 + 0);
                        T iii = src_leaf->at(i * 2 + 1, j * 2 + 1, k * 2 + 1);
                        T oio = src_leaf->at(i * 2 + 0, j * 2 + 1, k * 2 + 0);
                        T oii = src_leaf->at(i * 2 + 0, j * 2 + 1, k * 2 + 1);
                        T ooi = src_leaf->at(i * 2 + 0, j * 2 + 0, k * 2 + 1);
                        T ioi = src_leaf->at(i * 2 + 1, j * 2 + 0, k * 2 + 1);
                        T val = ooo + ioo + iio + iii + oio + oii + ooi + ioi;
                        leaf->at(
                            ii % 2 * (N / 2) + i,
                            jj % 2 * (N / 2) + j,
                            kk % 2 * (N / 2) + k) = val / 8;
                    }
                }
            }
        }
    }

    void prolongate(RBGrid const &src) {
        #pragma omp parallel for
        for (auto [src_leaf, ii, jj, kk]: src.get_leaves()) {
            for (long kb = 0; kb < 2; kb++) {
                for (long jb = 0; jb < 2; jb++) {
                    for (long ib = 0; ib < 2; ib++) {
                        auto *leaf = this->leaf_at(
                            ii * 2 + ib, jj * 2 + jb, kk * 2 + kb);
                        for (long k = 0; k < N; k++) {
                            for (long j = 0; j < N; j++) {
                                for (long i = 0; i < N; i++) {
                                    long ia = ib * (N / 2) + i / 2;
                                    long ja = jb * (N / 2) + j / 2;
                                    long ka = kb * (N / 2) + k / 2;
                                    T val = src_leaf->at(ia, ja, ka);
                                    leaf->at(i, j, k) = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};


#if 0
template <long N, class T>
void cgstep(RBGrid<N, T> &v, RBGrid<N, T> const &f, long times) {
    RBGrid<N, T> d;
    RBGrid<N, T> r;

    #pragma omp parallel for
    for (long _ = 0; _ < N * N * N; _++) {
        auto [i, j, k] = unlinearize(N, _);
        d.at(i, j, k) = f.at(i, j, k) - v.around_at(i, j, k);
    }
    r.copy(d);

    for (int t = 0; t < times; t++) {
        T dAd(0);
        #pragma omp parallel for reduction(+:dAd)
        for (long _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            dAd += d.around_at(i, j, k) * d.at(i, j, k);
        }

        T alpha = r.normsqr() / (dAd + 1e-6);

        T beta(0);
        #pragma omp parallel for reduction(+:beta)
        for (long _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            v.at(i, j, k) += alpha * d.at(i, j, k);
            r.at(i, j, k) -= alpha * d.around_at(i, j, k);
            T rr = r.at(i, j, k);
            beta += rr * rr;
        }
        beta /= (alpha + 1e-6) * dAd;

        #pragma omp parallel for
        for (long _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            d.at(i, j, k) = d.at(i, j, k) * beta + r.at(i, j, k);
        }
    }
}
#endif


template <long N, class T>
void vcycle(RBGrid<T> &v, RBGrid<T> const &f, long times) {
    v.smooth(f, times);

    if constexpr (N > 0) {
        RBGrid<T> r;
        v.residual(r, f);

        RBGrid<T> e2;
        RBGrid<T> r2;
        r2.restrict(r);

        vcycle<N - 1, T>(e2, r2, times << 1);

        RBGrid<T> e;
        e.prolongate(e2);
        v.add(e);
        v.smooth(f, times);
    }
}


int main(void)
{
    RBGrid<float> v;
    RBGrid<float> f;
    RBGrid<float> r;

    for (long k = 256-16; k < 256+16; k++) {
        for (long j = 256-16; j < 256+16; j++) {
            for (long i = 256-16; i < 256+16; i++) {
                //cout << i << ',' << j << ',' << k << endl;
                (void)f.leaf_at(i, j, k);
            }
        }
    }

    f.leaf_at(256, 256, 256)->at(4, 4, 4) = 32;
    v.copy_topology(f);
    show(v.leaf_count());
    show(f.leaf_count());

    auto t0 = std::chrono::steady_clock::now();
    v.smooth(f, 80);
    //vcycle<3>(v, f, 40);
    //cgstep(v, f, 20);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;

    v.residual(r, f);
    cout << std::sqrt(r.normsqr()) << endl;
    return 0;
}
