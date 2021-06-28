#pragma once
#include <vector>

#include "zensim/math/Vec.h"

namespace zs {

  template <typename Tn, auto dim> vec<Tn, dim> excl_suffix_prod(const vec<Tn, dim> &dims) {
    vec<Tn, dim> ret{};
    ret[dim - 1] = (Tn)1;
    for (int d = dim - 2; d >= 0; --d) ret[d] = ret[d + 1] * dims[d + 1];
    return ret;
  }

  template <typename T, typename Tn, Tn dim> struct DenseGrid {
    using IV = vec<Tn, dim>;

    Tn offset(const vec<Tn, dim> &indices) const noexcept {
      Tn ret{0};
      for (int d = 0; d < dim; ++d) ret += indices[d] * weights[d];
      return ret;
    }

    DenseGrid(const IV &dimsIn, T val) noexcept
        : weights{excl_suffix_prod(dimsIn)}, dims{dimsIn}, grid(weights.prod(), val) {}
    DenseGrid(IV &&dimsIn, T val) noexcept
        : weights{excl_suffix_prod(dimsIn)}, dims{dimsIn}, grid(weights.prod(), val) {}

    T operator()(const IV &indices) const { return grid[offset(indices)]; }
    T &operator()(const IV &indices) { return grid[offset(indices)]; }

    Tn domain(std::size_t d) const noexcept { return dims(d); }

    IV weights{};
    IV dims{};
    std::vector<T> grid{};
  };

}  // namespace zs
