#pragma once

#include <array>
#include "schedule.h"

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

    [[nodiscard]] static Qulong linearize(Quint3 coor) {
        return dot((coor + N) % N, Qulong3(1, N, N * N));
    }

    [[nodiscard]] static Quint3 delinearize(Qulong i) {
        return {i % N, (i / N) % N, (i / N) / N};
    }

    [[nodiscard]] T &operator()(Quint3 coor) {
        Qulong i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] T const &operator()(Quint3 coor) const {
        Qulong i = linearize(coor);
        return m_data[i];
    }

    [[nodiscard]] decltype(auto) operator()(Quint x, Quint y, Quint z) {
        return operator()({x, y, z});
    }

    [[nodiscard]] decltype(auto) operator()(Quint x, Quint y, Quint z) const {
        return operator()({x, y, z});
    }

    template <class Pol, class F>
    void foreach(Pol const &pol, F const &func) {
        pol->range_for(0, N * N * N, [&] (Qulong i) {
            Quint3 coor = delinearize(i);
            func(coor, m_data[i]);
        });
    }
};

}
