#include <cstring>
#include <cstdio>
#include "vec.h"
#include "timer.h"
#include "ndgrid.h"

using namespace zinc;

#define range(x, x0, x1) (uint32_t x = x0; x < x1; x++)

template <size_t T, size_t N>
void smooth(NDGrid<N> &v, NDGrid<N> const &f) {
    ZINC_PRETTY_TIMER;
    for range(phase, 0, T) {
#pragma omp parallel for
        for range(z, 0, N) {
            for range(y, 0, N) {
                for range(x, 0, N) {
                    if ((x + y + z) % 2 != phase % 2)
                        continue;
                    v(x, y, z) = (
                          f(x, y, z)
                        + v(x+1, y, z)
                        + v(x, y+1, z)
                        + v(x, y, z+1)
                        + v(x-1, y, z)
                        + v(x, y-1, z)
                        + v(x, y, z-1)
                        ) / 6;
                }
            }
        }
    }
}

template <size_t N>
void residual(NDGrid<N> &r, NDGrid<N> const &v, NDGrid<N> const &f) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                r(x, y, z) = (
                      f(x, y, z)
                    + v(x+1, y, z)
                    + v(x, y+1, z)
                    + v(x, y, z+1)
                    + v(x-1, y, z)
                    + v(x, y-1, z)
                    + v(x, y, z-1)
                    - v(x, y, z) * 6
                    );
            }
        }
    }
}

template <size_t N>
[[nodiscard]] float loss(NDGrid<N> const &v, NDGrid<N> const &f) {
    float res = 0.0f;
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                float val = f(x, y, z)
                    + v(x+1, y, z)
                    + v(x, y+1, z)
                    + v(x, y, z+1)
                    + v(x-1, y, z)
                    + v(x, y-1, z)
                    + v(x, y, z-1)
                    - v(x, y, z) * 6;
                res = fmaxf(res, fabsf(val));
                //res += val * val;
            }
        }
    }
    return res;
}

template <size_t N>
void restrict(NDGrid<N/2> &w, NDGrid<N> const &v) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N/2) {
        for range(y, 0, N/2) {
            for range(x, 0, N/2) {
                w(x, y, z) = (
                      v(x*2, y*2, z*2)
                    + v(x*2+1, y*2, z*2)
                    + v(x*2, y*2+1, z*2)
                    + v(x*2+1, y*2+1, z*2)
                    + v(x*2, y*2, z*2+1)
                    + v(x*2+1, y*2, z*2+1)
                    + v(x*2, y*2+1, z*2+1)
                    + v(x*2+1, y*2+1, z*2+1)
                    );
            }
        }
    }
}

template <size_t N>
void prolongate(NDGrid<N*2> &w, NDGrid<N> const &v) {
    ZINC_PRETTY_TIMER;
#pragma omp parallel for
    for range(z, 0, N*2) {
        for range(y, 0, N*2) {
            for range(x, 0, N*2) {
                w(x, y, z) += v(x/2, y/2, z/2);
            }
        }
    }
}

template <size_t T, size_t N>
void vcycle(NDGrid<N> &v, NDGrid<N> const &f) {
    if constexpr (N <= T) {
        smooth<T>(v, f);

    } else {
        smooth<T>(v, f);

        NDGrid<N> r;
        residual(r, v, f);

        NDGrid<N/2> r2;
        restrict(r2, r);

        NDGrid<N/2> e2;
        vcycle<T>(e2, r2);

        prolongate(v, e2);
        smooth<T>(v, f);
    }
}

template <size_t N>
void zeroinit(NDGrid<N> &v) {
    ZINC_PRETTY_TIMER;
    std::memset(v.m_data, 0, N * N * N * sizeof(float));
    std::memset(v.m_mask, 0, N * N * N / 8 * sizeof(uint8_t));
}

int main() {
    constexpr size_t N = 64;
    NDGrid<N> v, f;
    zeroinit(v);
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                f(x, y, z) = x == N/6 && y == N/2 && z == N/4 ? 100.0f : 0.0f;
            }
        }
    }
    printf("%f\n", loss(v, f));
    vcycle<16>(v, f);
    printf("%f\n", loss(v, f));
}
