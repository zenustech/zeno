#pragma once
#include <iomanip>
#include <iostream>
#include <vector>

namespace alpaca {

namespace detail {

template <typename Container> static inline void print_bytes(Container &bytes) {
  std::ios_base::fmtflags f(std::cout.flags());

  std::cout << "bytes[" << bytes.size() << "]:\n  ";

  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto &b = bytes[i];
    std::cout << std::hex << "0x" << std::setfill('0') << std::setw(2) << (int)b
              << " ";
    if (i > 0 && (i + 1) % 8 == 0) {
      std::cout << "\n  ";
    }
  }
  std::cout << "\n";

  std::cout.flags(f);
}

} // namespace detail

} // namespace alpaca