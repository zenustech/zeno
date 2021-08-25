#pragma once

#include "AutoInit.h"

namespace fdb {

template <class T, size_t N>
struct Dense {
    AutoInit<T> m_data[N * N * N];

    Dense() = default;
    ~Dense() = default;
    Dense(Dense const &) = default;
    Dense &operator=(Dense const &) = default;
    Dense(Dense &&) = default;
    Dense &operator=(Dense &&) = default;

    [[nodiscard]] static Qulong linearize(Qint3 coor) {
        //return dot(clamp(coor, 0, N-1), vec3L(1, N, N * N));
        return dot((coor + N) % N, vec3L(1, N, N * N));
        /*coor += N;
        Qulong i = dot((coor / 8) % (N/8), vec3L(1, N/8, N/8 * N/8));
        Qulong j = dot(coor % 8, vec3L(1, 8, 8 * 8));
        return 8*8*8 * i + j;*/
    }

    [[nodiscard]] T &operator()(Qint3 coor) {
        Qulong i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] T const &operator()(Qint3 coor) const {
        Qulong i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) operator()(Quint x, Quint y, Quint z) {
        return operator()({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(Quint x, Quint y, Quint z) const {
        return operator()({x, y, z});
    }
};

}
