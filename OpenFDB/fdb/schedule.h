#pragma once

#include <utility>
#include "vec.h"

namespace fdb {

struct Serial {
    template <class F, class T>
    void range_for(T num, F const &func) const {
        for (T i = 0; i < num; i++) {
            func(i);
        }
    }
};

struct Parallel {
    template <class F, class T>
    void range_for(T num, F const &func) const {
        #pragma omp parallel for
        for (T i = 0; i < num; i++) {
            func(i);
        }
    }
};

template <class Pol, class F, class T>
void range_for(Pol const &pol, T start, T stop, F const &func) {
    auto dist = stop - start;
    pol.range_for(dist, [&] (auto n) {
        func(start + n);
    });
}

template <class Pol, class F, class T, size_t N>
void ndrange_for(Pol const &pol, vec<T, N> start, vec<T, N> stop, F const &func) {
    auto dist = stop - start;
    auto dist_prod = dist[0];
    for (size_t i = 1; i < N; i++) {
        dist_prod *= dist[i];
    }
    pol.range_for(dist_prod, [&] (auto n) {
        vec<T, N> offs;
        for (size_t i = 0; i < N; i++) {
            offs[i] = n % dist[i];
            if (i != N - 1) n /= dist[i];
        }
        func(start + offs);
    });
}

}
