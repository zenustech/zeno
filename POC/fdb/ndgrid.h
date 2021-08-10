#pragma once

#include <cstdint>
#include "vec.h"

namespace zinc {

template <size_t N>
struct NDGrid {
    float *m_data = new float[N * N * N];

    NDGrid() = default;
    ~NDGrid() {
        delete[] m_data;
    }

    NDGrid(NDGrid const &) = delete;
    NDGrid &operator=(NDGrid const &) = delete;

    [[nodiscard]] static uintptr_t linearize(vec3I coor) {
        return dot(clamp(coor, 0, N-1), vec3L(1, N, N * N));
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

    [[nodiscard]] decltype(auto) operator()(vec3I &&coor) {
        return at(std::forward<vec3I>(coor));
    }

    [[nodiscard]] decltype(auto) operator()(vec3I &&coor) const {
        return at(std::forward<vec3I>(coor));
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) {
        return operator()({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(uint32_t x, uint32_t y, uint32_t z) const {
        return operator()({x, y, z});
    }
};

template <size_t N>
struct NDBitmask {
    uint8_t *m_mask = new uint8_t[N * N * N / 8];

    NDBitmask() = default;
    ~NDBitmask() {
        delete[] m_mask;
    }

    NDBitmask(NDBitmask const &) = delete;
    NDBitmask &operator=(NDBitmask const &) = delete;

    [[nodiscard]] static uintptr_t linearize(vec3I coor) {
        return dot(clamp(coor, 0, N-1), vec3L(1, N, N * N));
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
};

}
