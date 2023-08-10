#pragma once
#include <array>
#include <fstream>
#include <system_error>
#include <vector>

namespace alpaca {

namespace detail {

static inline void append(const uint8_t &value, std::vector<uint8_t> &container,
                          std::size_t &index) {
  container.push_back(value);
  index += 1;
}

template <std::size_t N>
void append(const uint8_t &value, std::array<uint8_t, N> &container,
            std::size_t &index) {
  container[index++] = value;
}

static inline void append(const uint8_t &value, uint8_t container[],
                          std::size_t &index) {
  container[index++] = value;
}

static inline void append(const uint8_t &value, std::ofstream &container,
                          std::size_t &index) {
  container << value;
  index += 1;
}

} // namespace detail

} // namespace alpaca
