#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>

using std::cout;
using std::endl;


inline std::tuple<size_t, size_t, size_t> unlinearize(size_t N, size_t idx) {
    return std::make_tuple<size_t, size_t, size_t>(
        idx % N, idx / N % N, idx / N / N);
}


template <class Leaf>
struct VDBGrid {
    struct InternalNode {
        Leaf *m[16 * 16 * 16];

        InternalNode() {
            std::memset(m, 0, sizeof(m));
        }

        ~InternalNode() {
            for (size_t i = 0; i < 16 * 16 * 16; i++) {
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
        for (size_t i = 0; i < 32 * 32 * 32; i++) {
            if (m[i]) {
                delete m[i];
                m[i] = nullptr;
            }
        }
    }

    Leaf *leaf_at(size_t i, size_t j, size_t k) {
        size_t ii = i / 16;
        size_t jj = j / 16;
        size_t kk = k / 16;
        size_t mm = ii + jj * (32 + kk * 32);
        InternalNode *&n = m[mm];
        if (!n) {
            n = new InternalNode();
        }
        ii = i % 16;
        jj = j % 16;
        kk = k % 16;
        mm = ii + jj * (16 + kk * 16);
        Leaf *&p = n->m[mm];
        if (!p) {
            p = new Leaf();
        }
        return p;
    }

    Leaf *ro_leaf_at(size_t i, size_t j, size_t k) const {
        size_t ii = i / 16;
        size_t jj = j / 16;
        size_t kk = k / 16;
        size_t mm = ii + jj * (32 + kk * 32);
        InternalNode *n = m[mm];
        if (!n) {
            return nullptr;
        }
        ii = i % 16;
        jj = j % 16;
        kk = k % 16;
        mm = ii + jj * (16 + kk * 16);
        Leaf *p = n->m[mm];
        return p;
    }

    template <class Func>
    void foreach_leaf(Func const &callback) const {
        for (size_t i = 0; i < 32 * 32 * 32; i++) {
            InternalNode *n = m[i];
            if (!n) continue;
            for (size_t j = 0; j < 16 * 16 * 16; j++) {
                Leaf *p = n->m[j];
                if (!p) continue;
                size_t k = i * 32 * 32 * 32 + j;
                size_t ii = k % 512;
                size_t jj = k / 512 % 512;
                size_t kk = k / 512 / 512;
                callback(p, ii, jj, kk);
            }
        }
    }
};


template <class T>
struct BoundaryLeaf {
    T m[10 * 10 * 10];

    BoundaryLeaf() {
        std::memset(m, 0, sizeof(m));
    }

    inline T &at(size_t i, size_t j, size_t k) {
        return m[linearize(i, j, k)];
    }

    inline T const &at(size_t i, size_t j, size_t k) const {
        return m[linearize(i, j, k)];
    }

    inline size_t linearize(size_t i, size_t j, size_t k) const {
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
    T m[8 * 8 * 8];

    SimpleLeaf() {
        std::memset(m, 0, sizeof(T) * 8 * 8 * 8);
    }

    inline T &at(size_t i, size_t j, size_t k) {
        return m[linearize(i, j, k)];
    }

    inline T const &at(size_t i, size_t j, size_t k) const {
        return m[linearize(i, j, k)];
    }

    inline size_t linearize(size_t i, size_t j, size_t k) const {
        return i + 8 * (j + 8 * k);
    }
};


template <class T>
struct RBGrid : VDBGrid<BoundaryLeaf<T>> {
    static constexpr size_t N = 8;

    void smooth(RBGrid const &rhs, size_t times) {
        //sync_boundaries();
        for (size_t t = 0; t < times; t++) {
            this->foreach_leaf([&] (auto *leaf, size_t ii, size_t jj, size_t kk) {
                auto *rhs_leaf = rhs.ro_leaf_at(ii, jj, kk);
                #pragma omp parallel for
                for (size_t k = 0; k < N; k++) {
                    for (size_t j = 0; j < N; j++) {
                        for (size_t i = (j + k + t) % 2; i < N; i += 2) {
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
            });
        }
    }

    T normsqr() const {
        T res(0);
        this->foreach_leaf([&] (auto *leaf, size_t ii, size_t jj, size_t kk) {
            #pragma omp parallel for reduction(+:res)
            for (size_t _ = 0; _ < N * N * N; _++) {
                auto [i, j, k] = unlinearize(N, _);
                T val = leaf->at(i, j, k);
                res += val * val;
            }
        });
        return res;
    }

    void residual(RBGrid &out, RBGrid const &rhs) const {
        this->foreach_leaf([&] (auto *leaf, size_t ii, size_t jj, size_t kk) {
            auto *out_leaf = out.leaf_at(ii, jj, kk);
            auto *rhs_leaf = rhs.ro_leaf_at(ii, jj, kk);
            #pragma omp parallel for
            for (size_t _ = 0; _ < N * N * N; _++) {
                auto [i, j, k] = unlinearize(N, _);
                T val = rhs_leaf->at(i, j, k) - leaf->around_at(i, j, k);
                out_leaf->at(i, j, k) = val;
            }
        });
    }

    void add(RBGrid const &src) {
        src.foreach_leaf([&] (auto *src_leaf, size_t ii, size_t jj, size_t kk) {
            auto *leaf = this->leaf_at(ii / 2, jj / 2, kk / 2);
            for (size_t k = 0; k < N; k++) {
                for (size_t j = 0; j < N; j++) {
                    for (size_t i = 0; i < N; i++) {
                        leaf->at(i, j, k) = src_leaf->at(i, j, k);
                    }
                }
            }
        });
    }

    void restrict(RBGrid const &src) {
        src.foreach_leaf([&] (auto *src_leaf, size_t ii, size_t jj, size_t kk) {
            auto *leaf = this->leaf_at(ii / 2, jj / 2, kk / 2);
            for (size_t k = 0; k < N / 2; k++) {
                for (size_t j = 0; j < N / 2; j++) {
                    for (size_t i = 0; i < N / 2; i++) {
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
        });
    }

    void prolongate(RBGrid const &src) {
        src.foreach_leaf([&] (auto *src_leaf, size_t ii, size_t jj, size_t kk) {
            for (size_t kb = 0; kb < 2; kb++) {
                for (size_t jb = 0; jb < 2; jb++) {
                    for (size_t ib = 0; ib < 2; ib++) {
                        auto *leaf = this->leaf_at(
                            ii * 2 + ib, jj * 2 + jb, kk * 2 + kb);
                        for (size_t k = 0; k < N; k++) {
                            for (size_t j = 0; j < N; j++) {
                                for (size_t i = 0; i < N; i++) {
                                    size_t ia = ib * (N / 2) + i / 2;
                                    size_t ja = jb * (N / 2) + j / 2;
                                    size_t ka = kb * (N / 2) + k / 2;
                                    T val = src_leaf->at(ia, ja, ka);
                                    leaf->at(i, j, k) = val;
                                }
                            }
                        }
                    }
                }
            }
        });
    }
};


#if 0
template <size_t N, class T>
struct RBGrid {
    T *m;

    RBGrid() {
        size_t size = (N + 2) * (N + 2) * (N + 2);
        m = new T[size];
        std::memset(m, 0, sizeof(T) * size);
    }

    ~RBGrid() {
        delete m;
        m = nullptr;
    }

    inline T &at(size_t i, size_t j, size_t k) {
        return m[linearize(i, j, k)];
    }

    inline T const &at(size_t i, size_t j, size_t k) const {
        return m[linearize(i, j, k)];
    }

    inline size_t linearize(size_t i, size_t j, size_t k) const {
        return i + 1 + (N + 2) * (j + 1 + (N + 2) * (k + 1));
    }

    void smooth(RBGrid const &rhs, size_t times) {
        for (size_t t = 0; t < times; t++) {
            #pragma omp parallel for
            for (size_t k = 0; k < N; k++) {
                for (size_t j = 0; j < N; j++) {
                    for (size_t i = (j + k + t) % 2; i < N; i += 2) {
                        T x1 = at(i - 1, j, k);
                        T x2 = at(i + 1, j, k);
                        T y1 = at(i, j - 1, k);
                        T y2 = at(i, j + 1, k);
                        T z1 = at(i, j, k - 1);
                        T z2 = at(i, j, k + 1);
                        T f = rhs.at(i, j, k);
                        T val = x1 + x2 + y1 + y2 + z1 + z2 + f;
                        at(i, j, k) = val / 6;
                    }
                }
            }
        }
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

    void residual(RBGrid &out, RBGrid const &rhs) const {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            T val = rhs.at(i, j, k) - around_at(i, j, k);
            out.at(i, j, k) = val;
        }
    }

    void restrict(RBGrid<N * 2, T> const &src) {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            T ooo = src.at(i * 2 + 0, j * 2 + 0, k * 2 + 0);
            T ioo = src.at(i * 2 + 1, j * 2 + 0, k * 2 + 0);
            T iio = src.at(i * 2 + 1, j * 2 + 1, k * 2 + 0);
            T iii = src.at(i * 2 + 1, j * 2 + 1, k * 2 + 1);
            T oio = src.at(i * 2 + 0, j * 2 + 1, k * 2 + 0);
            T oii = src.at(i * 2 + 0, j * 2 + 1, k * 2 + 1);
            T ooi = src.at(i * 2 + 0, j * 2 + 0, k * 2 + 1);
            T ioi = src.at(i * 2 + 1, j * 2 + 0, k * 2 + 1);
            T val = ooo + ioo + iio + iii + oio + oii + ooi + ioi;
            at(i, j, k) = val / 8;
        }
    }

    void prolongate(RBGrid<N / 2, T> const &src) {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            at(i, j, k) = src.at(i / 2, j / 2, k / 2);
        }
    }

    void add(RBGrid const &src) {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            at(i, j, k) += src.at(i, j, k);
        }
    }

    void copy(RBGrid const &src) {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            at(i, j, k) = src.at(i, j, k);
        }
    }

    void fill(T val) {
        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            at(i, j, k) = val;
        }
    }

    T normsqr() const {
        T res(0);
        #pragma omp parallel for reduction(+:res)
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            T val = at(i, j, k);
            res += val * val;
        }
        return res;
    }
};
#endif


#if 0
template <size_t N, class T>
void cgstep(RBGrid<N, T> &v, RBGrid<N, T> const &f, size_t times) {
    RBGrid<N, T> d;
    RBGrid<N, T> r;

    #pragma omp parallel for
    for (size_t _ = 0; _ < N * N * N; _++) {
        auto [i, j, k] = unlinearize(N, _);
        d.at(i, j, k) = f.at(i, j, k) - v.around_at(i, j, k);
    }
    r.copy(d);

    for (int t = 0; t < times; t++) {
        T dAd(0);
        #pragma omp parallel for reduction(+:dAd)
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            dAd += d.around_at(i, j, k) * d.at(i, j, k);
        }

        T alpha = r.normsqr() / (dAd + 1e-6);

        T beta(0);
        #pragma omp parallel for reduction(+:beta)
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            v.at(i, j, k) += alpha * d.at(i, j, k);
            r.at(i, j, k) -= alpha * d.around_at(i, j, k);
            T rr = r.at(i, j, k);
            beta += rr * rr;
        }
        beta /= (alpha + 1e-6) * dAd;

        #pragma omp parallel for
        for (size_t _ = 0; _ < N * N * N; _++) {
            auto [i, j, k] = unlinearize(N, _);
            d.at(i, j, k) = d.at(i, j, k) * beta + r.at(i, j, k);
        }
    }
}
#endif


template <size_t N, class T>
void vcycle(RBGrid<T> &v, RBGrid<T> const &f, size_t times) {
    v.smooth(f, times);

    if constexpr (N > 32) {
        RBGrid<T> r;
        v.residual(r, f);

        RBGrid<T> e2;
        RBGrid<T> r2;
        r2.restrict(r);

        vcycle<N / 2, T>(e2, r2, times << 1);

        RBGrid<T> e;
        e.prolongate(e2);
        v.add(e);
        v.smooth(f, times);
    }
}


int main(void)
{
    constexpr size_t N = 128;
    RBGrid<float> v;
    RBGrid<float> f;
    RBGrid<float> r;
    v.leaf_at(2048, 2048, 2048);
    f.leaf_at(2048, 2048, 2048)->at(N / 2, N / 2, N / 2) = 32;

    auto t0 = std::chrono::steady_clock::now();
    vcycle<N>(v, f, 80);
    //cgstep<N>(v, f, 40);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;

    v.residual(r, f);
    cout << std::sqrt(r.normsqr()) << endl;
}
