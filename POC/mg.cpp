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


template <size_t N, class T>
struct RBGrid {
    T *m;

    RBGrid() {
        m = new T[N * N * N];
        std::memset(m, 0, sizeof(T) * N * N * N);
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
        //static_assert(N % 2 == 0);
        //k = k / 2 + (N / 2) * ((i + j + k) % 2);
        return i + N * (j + k * N);
    }

    void smooth(RBGrid const &rhs, size_t times) {
        for (size_t t = 0; t < (times + 1) / 2 * 2; t++) {
            #pragma omp parallel for schedule(static)
            for (size_t k = 1; k < N - 1; k++) {
                for (size_t j = 1; j < N - 1; j++) {
                    for (size_t i = 1 + (j + k + t) % 2; i < N - 1; i += 2) {
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
        T x1 = i > 0 ? at(i - 1, j, k) : T(0);
        T x2 = i < N - 1 ? at(i + 1, j, k) : T(0);
        T y1 = j > 0 ? at(i, j - 1, k) : T(0);
        T y2 = j < N - 1 ? at(i, j + 1, k) : T(0);
        T z1 = k > 0 ? at(i, j, k - 1) : T(0);
        T z2 = k < N - 1 ? at(i, j, k + 1) : T(0);
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


template <size_t N, class T>
void vcycle(RBGrid<N, T> &v, RBGrid<N, T> const &f, size_t times) {
    v.smooth(f, times);

    if constexpr (N > 32) {
        RBGrid<N, T> r;
        v.residual(r, f);

        RBGrid<N / 2, T> e2;
        RBGrid<N / 2, T> r2;
        r2.restrict(r);

        vcycle<N / 2, T>(e2, r2, times << 1);

        RBGrid<N, T> e;
        e.prolongate(e2);
        v.add(e);
        v.smooth(f, times);
    }
}


int main(void)
{
    constexpr size_t N = 128;
    RBGrid<N, float> v;
    RBGrid<N, float> f;
    RBGrid<N, float> r;
    f.at(N / 2, N / 2, N / 2) = 32;

    auto t0 = std::chrono::steady_clock::now();
    vcycle(v, f, 80);
    cgstep(v, f, 40);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;

    v.residual(r, f);
    cout << std::sqrt(r.normsqr()) << endl;
}
