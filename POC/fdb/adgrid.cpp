#include <functional>
#include <cstring>
#include <cstdio>
#include "vec.h"

using namespace fdb;

template <size_t N>
struct NDGrid {
    float m_data[N * N * N];
    bool m_mask[N * N * N];

    auto &at(vec3L coor) {
        uintptr_t i = dot(coor, vec3i(1, N, N * N);
        return m_data[i];
    }
};


