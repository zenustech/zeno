#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <benchmark/benchmark.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range3d.h>
#include <x86intrin.h>
#include "mtprint.h"
#include <array>
#include <atomic>

constexpr size_t n = 512;

std::vector<float> a(n * n * n);
std::vector<float> b(n * n * n);

static void copy() {
#if 1
    tbb::parallel_for(tbb::blocked_range3d<size_t>(0, n, 0, n, 0, n),
    [&] (tbb::blocked_range3d<size_t> const &r) {
        for (size_t z = r.pages().begin(); z < r.pages().end(); z++) {
            for (size_t y = r.cols().begin(); y < r.cols().end(); y++) {
                for (size_t x = r.rows().begin(); x < r.rows().end(); x++) {
                    b[(z * n + y) * n + x] = a[(z * n + y) * n + x];
                }
            }
        }
    });
#else
    std::memcpy(a.data(), b.data(), n * n * n * sizeof(float));
#endif
}

static void BM_copy(benchmark::State &bm) {
    for (auto _: bm) {
        copy();
    }
}
BENCHMARK(BM_copy);

static void jacobi() {
    tbb::parallel_for(tbb::blocked_range3d<size_t>(0, n, 0, n, 0, n),
    [&] (tbb::blocked_range3d<size_t> const &r) {
        for (size_t z = std::max<size_t>(1, r.pages().begin()); z < std::min(n - 2, r.pages().end()); z++) {
            for (size_t y = std::max<size_t>(1, r.cols().begin()); y < std::min(n - 2, r.cols().end()); y++) {
                for (size_t x = std::max<size_t>(1, r.rows().begin()); x < std::min(n - 2, r.rows().end()); x++) {
                    b[(z * n + y) * n + x] = -6 * a[(z * n + y) * n + x]
                        + a[(z * n + y) * n + x + 1]
                        + a[(z * n + y) * n + x - 1]
                        + a[(z * n + y) * n + x + n]
                        + a[(z * n + y) * n + x - n]
                        + a[(z * n + y) * n + x + n * n]
                        + a[(z * n + y) * n + x - n * n]
                        ;
                }
            }
        }
    });
}

static void BM_jacobi(benchmark::State &bm) {
    for (auto _: bm) {
        jacobi();
    }
}
BENCHMARK(BM_jacobi);

static void jacobi_grained() {
    constexpr size_t gs = 128;
    tbb::parallel_for(tbb::blocked_range3d<size_t>(0, n, gs, 0, n, gs, 0, n, gs),
    [&] (tbb::blocked_range3d<size_t> const &r) {
        for (size_t z = std::max<size_t>(1, r.pages().begin()); z < std::min(n - 2, r.pages().end()); z++) {
            for (size_t y = std::max<size_t>(1, r.cols().begin()); y < std::min(n - 2, r.cols().end()); y++) {
                for (size_t x = std::max<size_t>(1, r.rows().begin()); x < std::min(n - 2, r.rows().end()); x++) {
                    b[(z * n + y) * n + x] = -6 * a[(z * n + y) * n + x]
                        + a[(z * n + y) * n + x + 1]
                        + a[(z * n + y) * n + x - 1]
                        + a[(z * n + y) * n + x + n]
                        + a[(z * n + y) * n + x - n]
                        + a[(z * n + y) * n + x + n * n]
                        + a[(z * n + y) * n + x - n * n]
                        ;
                }
            }
        }
    }, tbb::simple_partitioner{});
}

static void BM_jacobi_grained(benchmark::State &bm) {
    for (auto _: bm) {
        jacobi_grained();
    }
}
BENCHMARK(BM_jacobi_grained);

static uint64_t mortondec0(uint64_t x) {
    x = x & 0x5555555555555555ul;
    x = (x | (x >> 1)) & 0x3333333333333333ull;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FFull;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFFull;
    x = (x | (x >> 16)) & 0xFFFFFFFFFFFFFFFFull;
    return x;
}

static uint32_t mortondec1(uint32_t x) {
    x &= 0x09249249;
    x = (x ^ (x >>  2)) & 0x030c30c3;
    x = (x ^ (x >>  4)) & 0x0300f00f;
    x = (x ^ (x >>  8)) & 0xff0000ff;
    x = (x ^ (x >> 16)) & 0x000003ff;
    return x;
}

static void jacobi_morton() {
    constexpr size_t bs = 128;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n / bs * n / bs * n / bs),
    [&] (tbb::blocked_range<size_t> const &ro) {
        for (size_t t = ro.begin(); t < ro.end(); t++) {
            size_t x0 = mortondec1(t) * bs;
            size_t y0 = mortondec1(t >> 1) * bs;
            size_t z0 = mortondec1(t >> 2) * bs;
            tbb::blocked_range3d<size_t> r(x0, x0 + bs, y0, y0 + bs, z0, z0 + bs);
            for (size_t z = std::max<size_t>(1, r.pages().begin()); z < std::min(n - 2, r.pages().end()); z++) {
                for (size_t y = std::max<size_t>(1, r.cols().begin()); y < std::min(n - 2, r.cols().end()); y++) {
                    for (size_t x = std::max<size_t>(1, r.rows().begin()); x < std::min(n - 2, r.rows().end()); x++) {
                        b[(z * n + y) * n + x] = -6 * a[(z * n + y) * n + x]
                            + a[(z * n + y) * n + x + 1]
                            + a[(z * n + y) * n + x - 1]
                            + a[(z * n + y) * n + x + n]
                            + a[(z * n + y) * n + x - n]
                            + a[(z * n + y) * n + x + n * n]
                            + a[(z * n + y) * n + x - n * n]
                            ;
                    }
                }
            }
        }
    });
}

static void BM_jacobi_morton(benchmark::State &bm) {
    for (auto _: bm) {
        jacobi_morton();
    }
}
BENCHMARK(BM_jacobi_morton);

BENCHMARK_MAIN();
