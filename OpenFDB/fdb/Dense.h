#pragma once

#include <array>
#include "policy.h"
#include "types.h"

namespace fdb {

template <class T, size_t N>
struct Dense {
    std::array<T, N * N * N> m_data;

    Dense() = default;
    ~Dense() = default;
    Dense(Dense const &) = default;
    Dense &operator=(Dense const &) = default;
    Dense(Dense &&) = default;
    Dense &operator=(Dense &&) = default;

    [[nodiscard]] static size_t linearize(vec<size_t, 3> coor) {
        return dot((coor + N) % N, vec<size_t, 3>(1, N, N * N));
    }

    [[nodiscard]] static vec<size_t, 3> delinearize(size_t i) {
        return vec<size_t, 3>(i % N, (i / N) % N, (i / N) / N);
    }

    [[nodiscard]] T &at(vec<size_t, 3> coor) {
        size_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] T const &at(vec<size_t, 3> coor) const {
        size_t i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) operator()(vec<size_t, 3> coor) {
        return at(coor);
    }

    [[nodiscard]] decltype(auto) operator()(vec<size_t, 3> coor) const {
        return at(coor);
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        range_for(pol, (size_t)0, N * N * N, [&] (size_t i) {
            vec<size_t, 3> coor = delinearize(i);
            func(coor, m_data[i]);
        });
    }
};

}
