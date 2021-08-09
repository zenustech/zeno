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

    static uintptr_t linearize(vec3I coor) {
        return dot(coor, vec3L(1, N, N * N));
    }

    bool is_active(vec3I coor) const {
        uintptr_t i = linearize(coor);
        return m_mask[i >> 3] & (1 << (i & 7));
    }

    void activate(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] |= 1 << (i & 7);
    }

    void deactivate(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] &= ~(1 << (i & 7));
    }

    auto const &at(vec3I coor) const {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    auto &at(vec3I coor) {
        uintptr_t i = linearize(coor);
        return m_data[i];
    }

    auto &aat(vec3I coor) {
        uintptr_t i = linearize(coor);
        m_mask[i >> 3] |= 1 << (i & 7);
        return m_data[i];
    }

    decltype(auto) operator()(vec3I &&coor) {
        return at(std::forward<vec3I>(coor));
    }

    decltype(auto) operator()(vec3I &&coor) const {
        return at(std::forward<vec3I>(coor));
    }

    decltype(auto) operator[](vec3I &&coor) {
        return aat(std::forward<vec3I>(coor));
    }
};

#define range(x, x0, x1) (uint32_t x = x0; x < x1; x++)

int main() {
    NDGrid<16> cu;
    NDGrid<32> xi;
    for range(z, 0, 16) {
        for range(y, 0, 16) {
            for range(x, 0, 16) {
                cu.aat({x, y, z}) = 1.0f;
            }
        }
    }
    for range(z, 14, 18) {
        for range(y, 14, 18) {
            for range(x, 14, 18) {
                xi.aat({x, y, z}) = 2.0f;
            }
        }
    }
}
