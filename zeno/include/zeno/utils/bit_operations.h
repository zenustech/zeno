#pragma once

#include <cstdint>
#include <utility>

namespace zeno {

template <class To, class T>
constexpr To const &bit_cast(T const &t) noexcept {
    return *reinterpret_cast<To const *>(std::addressof(t));
}

template <class To, class T>
constexpr To &bit_cast(T &t) noexcept {
    return *reinterpret_cast<To *>(std::addressof(t));
}

static constexpr std::uint8_t ceil_log2(std::size_t x) {
    const std::uint64_t t[6] = {
        0xFFFFFFFF00000000ull,
        0x00000000FFFF0000ull,
        0x000000000000FF00ull,
        0x00000000000000F0ull,
        0x000000000000000Cull,
        0x0000000000000002ull
    };
    int y = (((x & (x - 1)) == 0) ? 0 : 1);
    int j = 32;
    for (int i = 0; i < 6; i++) {
        int k = (((x & t[i]) == 0) ? 0 : j);
        y += k;
        x >>= k;
        j >>= 1;
    }
    return std::uint8_t(y);
}

}
