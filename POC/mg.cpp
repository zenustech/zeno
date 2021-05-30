#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>

using std::cout;
using std::endl;


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

    T &at(size_t i, size_t j, size_t k) {
        return m[linearize(i, j, k)];
    }

    T const &at(size_t i, size_t j, size_t k) const {
        return m[linearize(i, j, k)];
    }

    size_t linearize(size_t i, size_t j, size_t k) const {
        //static_assert(N % 2 == 0);
        //k = k / 2 + (N / 2) * ((i + j + k) % 2);
        return i + j * N + k * (N * N);
    }

    void smooth(RBGrid const &rhs, size_t times) {
        for (size_t t = 0; t < (times + 1) / 2 * 2; t++) {
            for (size_t k = 1; k < N - 1; k++) {
                for (size_t j = 1; j < N - 1; j++) {
                    for (size_t i = 1; i < N - 1; i++) {
                        if ((i + j + k) % 2 == t % 2) {
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
    }

    void residual(RBGrid &out, RBGrid const &rhs) {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    T x1 = i > 0 ? at(i - 1, j, k) : T(0);
                    T x2 = i < N - 1 ? at(i + 1, j, k) : T(0);
                    T y1 = j > 0 ? at(i, j - 1, k) : T(0);
                    T y2 = j < N - 1 ? at(i, j + 1, k) : T(0);
                    T z1 = k > 0 ? at(i, j, k - 1) : T(0);
                    T z2 = k < N - 1 ? at(i, j, k + 1) : T(0);
                    T f = rhs.at(i, j, k);
                    T c = at(i, j, k);
                    T val = x1 + x2 + y1 + y2 + z1 + z2 + f - 6 * c;
                    out.at(i, j, k) = val;
                }
            }
        }
    }

    void restrict(RBGrid<N * 2, T> const &src) {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
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
        }
    }

    void prolongate(RBGrid<N / 2, T> const &src) {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    at(i, j, k) = src.at(i / 2, j / 2, k / 2);
                }
            }
        }
    }

    void add(RBGrid const &src) {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    at(i, j, k) += src.at(i / 2, j / 2, k / 2);
                }
            }
        }
    }

    T fill(T val) const {
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    at(i, j, k) = val;
                }
            }
        }
    }

    T norm() const {
        T res(0);
        for (size_t k = 0; k < N; k++) {
            for (size_t j = 0; j < N; j++) {
                for (size_t i = 0; i < N; i++) {
                    T val = at(i, j, k);
                    res += val * val;
                }
            }
        }
        return std::sqrt(res);
    }
};


template <size_t N, class T>
void vcycle(RBGrid<N, T> &v, RBGrid<N, T> const &f, size_t times, size_t lev = 0) {
    v.smooth(f, times << lev);

    if constexpr (N > 32) {
        RBGrid<N, T> r;
        v.residual(r, f);

        RBGrid<N / 2, T> e2;
        RBGrid<N / 2, T> r2;
        r2.restrict(r);

        vcycle<N / 2, T>(e2, r2, lev + 1);

        RBGrid<N, T> e;
        e.prolongate(e2);
        v.add(e);
        v.smooth(f, times << lev);
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
    vcycle(v, f, 64);
    //v.smooth(f, 128);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " ms" << endl;

    v.residual(r, f);
    cout << r.norm() << endl;
}
