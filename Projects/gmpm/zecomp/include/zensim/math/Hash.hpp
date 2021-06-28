#pragma once

namespace zs {

  //  on why XOR is not a good choice for hash-combining:
  //  https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
  //
  //  this is from boost
  //
  template <typename T> constexpr void hash_combine(std::size_t &seed, const T &val) {
    seed ^= (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
  }

/// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
#if 0
constexpr uint32_t hash(uint32_t x) noexcept {
  x += (x << 10u);
  x ^= (x >> 6u);
  x += (x << 3u);
  x ^= (x >> 11u);
  x += (x << 15u);
  return x;
}
#endif
  template <typename T> constexpr T hash(T x) noexcept {
    if constexpr (sizeof(T) == 4) {
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = ((x >> 16) ^ x) * 0x45d9f3b;
      x = (x >> 16) ^ x;
    } else if constexpr (sizeof(T) == 8) {
      x = (x ^ (x >> 30)) * uint64_t(0xbf58476d1ce4e5b9);
      x = (x ^ (x >> 27)) * uint64_t(0x94d049bb133111eb);
      x = x ^ (x >> 31);
    }
    return x;
  }
  template <typename T> constexpr T unhash(T x) noexcept {
    if constexpr (sizeof(T) == 4) {
      x = ((x >> 16) ^ x) * 0x119de1f3;
      x = ((x >> 16) ^ x) * 0x119de1f3;
      x = (x >> 16) ^ x;
    } else if constexpr (sizeof(T) == 8) {
      x = (x ^ (x >> 31) ^ (x >> 62)) * uint64_t(0x319642b2d24d8ec3);
      x = (x ^ (x >> 27) ^ (x >> 54)) * uint64_t(0x96de1b173f119089);
      x = x ^ (x >> 30) ^ (x >> 60);
    }
    return x;
  }

}  // namespace zs
