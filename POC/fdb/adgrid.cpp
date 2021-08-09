#include <functional>
#include <cstring>
#include <cstdio>
#include "vec.h"

using namespace fdb;

template <size_t N>
struct NDGrid {
    float *m_data = new float[N * N * N];
    uint8_t *m_mask = new uint8_t[N * N * N / 8];

    ~NDGrid() {
        delete[] m_data;
        delete[] m_mask;
    }

    NDGrid(NDGrid const &) = delete;
    NDGrid &operator=(NDGrid const &) = delete;

    bool is_active(vec3L coor) const {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        return m_mask[i >> 3] & (1 << (i & 7));
    }

    bool activate(vec3L coor) {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        m_mask[i >> 3] |= 1 << (i & 7);
    }

    bool deactivate(vec3L coor) {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        m_mask[i >> 3] &= ~(1 << (i & 7));
    }

    auto const &at(vec3L coor) const {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        return m_data[i];
    }

    auto &at(vec3L coor) {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        return m_data[i];
    }

    auto &activate_at(vec3L coor) {
        uintptr_t i = dot(coor, vec3i(1, N, N * N));
        m_mask[i >> 3] |= 1 << (i & 7);
        return m_data[i];
    }
};


