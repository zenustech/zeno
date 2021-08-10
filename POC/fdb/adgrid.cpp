#include <functional>
#include <cstring>
#include <cstdio>
#include "vec.h"

using namespace fdb;

template <size_t N>
struct NDGrid {
    float *m_data = new float[N * N * N];
    uint8_t *m_mask = new uint8_t[N * N * N / 8];

    NDGrid() {
        std::memset(m_data, 0, N * N * N * sizeof(float));
        std::memset(m_mask, 0, N * N * N / 8 * sizeof(uint8_t));
    }

    ~NDGrid() {
        delete[] m_data;
        delete[] m_mask;
    }

    NDGrid(NDGrid const &) = delete;
    NDGrid &operator=(NDGrid const &) = delete;

    [[nodiscard]] static uintptr_t linearize(vec3I coor) {
        return dot(coor, vec3L(1, N, N * N));
    }

    [[nodiscard]] bool is_active(vec3I coor) const {
        uintptr_t i = linearize(coor);
        return m_mask[i >> 3] & (1 << (i & 7));
    }

    [[nodiscard]] bool is_active(uint32_t x, uint32_t y, uint32_t z) {
        return is_active({x, y, z});
    }

    void activate(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] |= 1 << (i & 7);
    }

    void activate(uint32_t x, uint32_t y, uint32_t z) {
        return activate({x, y, z});
    }

    void deactivate(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] &= ~(1 << (i & 7));
    }

    void deactivate(uint32_t x, uint32_t y, uint32_t z) {
        return deactivate({x, y, z});
    }

    [[nodiscard]] auto const &at(vec3I coor) const {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] auto &at(vec3I coor) {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) at(uint32_t x, uint32_t y, uint32_t z) {
        return at({x, y, z});
    }

    [[nodiscard]] decltype(auto) at(uint32_t x, uint32_t y, uint32_t z) const {
        return at({x, y, z});
    }

    [[nodiscard]] auto &aat(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] |= 1 << (i & 7);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) aat(uint32_t x, uint32_t y, uint32_t z) {
        return aat({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(vec3I &&coor) {
        return at(std::forward<vec3I>(coor));
    }

    [[nodiscard]] decltype(auto) operator()(vec3I &&coor) const {
        return at(std::forward<vec3I>(coor));
    }

    [[nodiscard]] decltype(auto) operator[](vec3I &&coor) {
        return aat(std::forward<vec3I>(coor));
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) {
        return operator()({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) const {
        return operator()({x, y, z});
    }
};

#define range(x, x0, x1) (uint32_t x = x0; x < x1; x++)

int smooth_count;

template <size_t N>
void smooth(NDGrid<N> &v, NDGrid<N> const &f, int times = 4) {
    for range(phase, 0, times) {
        for range(z, 1, N-1) {
            for range(y, 1, N-1) {
                for range(x, 1, N-1) {
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
    smooth_count += times * N * N * N;
}

template <size_t N>
void residual(NDGrid<N> &r, NDGrid<N> const &v, NDGrid<N> const &f) {
    for range(z, 1, N-1) {
        for range(y, 1, N-1) {
            for range(x, 1, N-1) {
                r(x, y, z) = 0.5f * (
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
[[nodiscard]] float residual(NDGrid<N> const &v, NDGrid<N> const &f) {
    float res = 0.0f;
    for range(z, 1, N-1) {
        for range(y, 1, N-1) {
            for range(x, 1, N-1) {
                float val = f(x, y, z)
                    + v(x+1, y, z)
                    + v(x, y+1, z)
                    + v(x, y, z+1)
                    + v(x-1, y, z)
                    + v(x, y-1, z)
                    + v(x, y, z-1)
                    - v(x, y, z) * 6;
                res += val * val;
            }
        }
    }
    return res * 0.25f;
}

template <size_t N>
void restrict(NDGrid<N/2> &w, NDGrid<N> const &v) {
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
                    ) / 8;
            }
        }
    }
}

template <size_t N>
void prolongate(NDGrid<N*2> &w, NDGrid<N> const &v) {
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
    if constexpr (N <= 8) {
        smooth(v, f, T);

    } else {
        smooth(v, f, T);

        NDGrid<N> r;
        residual(r, v, f);

        NDGrid<N/2> r2;
        restrict(r2, r);

        NDGrid<N/2> e2;
        vcycle<T>(e2, r2);

        prolongate(v, e2);
        smooth(v, f, T);
    }
}

int main() {
    constexpr size_t N = 32;
    NDGrid<N> v, f;
    for range(z, 0, N) {
        for range(y, 0, N) {
            for range(x, 0, N) {
                f(x, y, z) = x == N/2 && y == N/2 && z == N/2 ? 10.0f : 0.0f;
            }
        }
    }
    printf("%f\n", residual(v, f));
    vcycle<8>(v, f);
    //smooth(v, f, 18);
    printf("%f\n", residual(v, f));
    printf("%d\n", smooth_count);
}
