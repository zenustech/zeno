#pragma once
#include <vector>

namespace zs {

  template <typename T, int dim, typename Tn = int, int dimE = dim + 1> struct Mesh {
    using Node = std::array<T, dim>;
    using Elem = std::array<Tn, dimE>;
    std::vector<Node> nodes;
    std::vector<Elem> elems;
  };

}  // namespace zs
