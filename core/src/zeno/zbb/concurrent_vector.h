// TODO: https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
#pragma once


#include <zeno/common.h>
#include <vector>
#include <array>
#include <mutex>
#include <new>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
struct concurrent_vector {
    using value_type = T;
    inline static constexpr std::uint8_t _kBase = 8;
    inline static constexpr std::uint8_t _kMax = 32;

    std::array<std::vector<T>, _kMax> _bins;
    std::uint8_t _binid{};
    std::mutex _mtx;

    concurrent_vector() = default;
    concurrent_vector(concurrent_vector const &other) = default;
    concurrent_vector(concurrent_vector &&) = default;
    concurrent_vector &operator=(concurrent_vector const &) = default;
    concurrent_vector &operator=(concurrent_vector &&) = default;
    ~concurrent_vector() = default;

    T *grow_twice() noexcept {
        std::lock_guard _(_mtx);
        std::uint8_t nbin = _binid++;
        auto &bin = _bins[nbin];
        bin.resize(1 << nbin);
        return bin.data();
    }

    [[nodiscard]] T &at(std::size_t off) noexcept {
        std::uint8_t pot = 0;
        for (std::size_t i = off >> _kBase; i; i >>= 1)
            pot++;
        return _bins[pot][off & (1 << pot + _kBase) - 1];
    }
};


}
ZENO_NAMESPACE_END
