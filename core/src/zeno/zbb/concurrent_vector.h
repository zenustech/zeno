// TODO: https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector
#pragma once


#include <zeno/common.h>
#include <vector>
#include <array>


ZENO_NAMESPACE_BEGIN
namespace zbb {


template <class T>
struct pot_scaled_vector {
    using value_type = T;
    inline static constexpr std::uint8_t _kMin = 8;
    inline static constexpr std::uint8_t _kMax = 32;

    std::array<std::vector<T>, _kMax> _bins;
    std::uint8_t _numbins{};

    pot_scaled_vector() = default;
    pot_scaled_vector(pot_scaled_vector const &other) = default;
    pot_scaled_vector(pot_scaled_vector &&) = default;
    pot_scaled_vector &operator=(pot_scaled_vector const &) = default;
    pot_scaled_vector &operator=(pot_scaled_vector &&) = default;
    ~pot_scaled_vector() = default;

    std::vector<T> &_grow_twice() noexcept {
        std::uint8_t nbin = _numbins++;
        std::vector<T> &bin = _bins[nbin];
        bin.resize(1 << nbin + _kMin);
        return bin;
    }

    std::vector<T> &grow(std::size_t ext) noexcept {
        std::uint8_t nbin = _numbins - 1;
        std::vector<T> &bin = _bins[nbin];
        std::size_t limit = 1 << _kMin << nbin;
        std::size_t curr = bin.size();
        if (curr + ext < limit) {
            bin.resize(curr + ext);
            return bin;
        } else {
            if (curr != limit)
                bin.resize(limit);
            ext -= limit - curr;
            _numbins = ++nbin;
            std::vector<T> &bin = _bins[nbin];
            bin.resize(ext);
            return bin;
        }
    }

    T &operator[](std::size_t off) noexcept {
        off += 1 << _kMin;
        std::uint8_t pot = 0;
        for (std::size_t i = off >> (1 + _kMin); i; i >>= 1)
            pot++;
        off -= 1 << _kMin << pot;
        return _bins[pot][off];
    }

    T const &operator[](std::size_t off) const noexcept {
        off += 1 << _kMin;
        std::uint8_t pot = 0;
        for (std::size_t i = off >> (1 + _kMin); i; i >>= 1)
            pot++;
        off -= 1 << _kMin << pot;
        return _bins[pot][off];
    }
};


}
ZENO_NAMESPACE_END
