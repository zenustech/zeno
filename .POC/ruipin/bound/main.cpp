#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <benchmark/benchmark.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <x86intrin.h>
#include "mtprint.h"
#include <array>
#include <atomic>

template <size_t kSize, size_t kOffset, class Value>
struct Grid {
    std::array<Value, (kSize + kOffset * 2) * (kSize + kOffset * 2)> mArr;

    constexpr Value &operator()(size_t i, size_t j) {
        return mArr[(j + kOffset) * (kSize + kOffset * 2) + (i + kOffset)];
    }
};

template <size_t kBlock, size_t kSize, size_t kOffset, class Value>
static void jacobi(Grid<kSize, kOffset, Value> &ogrid, Grid<kSize, kOffset, Value> &igrid) {
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, kSize, kBlock, 0, kSize, kBlock),
    [&] (tbb::blocked_range2d<size_t> const &r) {
        for (size_t y = r.cols().begin(); y < r.cols().end(); y++) {
            for (size_t x = r.rows().begin(); x < r.rows().end(); x++) {
                ogrid(x, y) = igrid(x + 1, y) + igrid(x - 1, y) + igrid(x, y + 1) + igrid(x, y - 1);
            }
        }
    });
}

static void BM_jacobi(benchmark::State &bm) {
    static Grid<4096, 0, float> g1, g2;
    for (auto _: bm) {
        jacobi<32>(g1, g2);
    }
}
BENCHMARK(BM_jacobi);

BENCHMARK_MAIN();
