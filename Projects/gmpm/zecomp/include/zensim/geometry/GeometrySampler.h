#pragma once
#include <array>
#include <vector>

#include "LevelSet.h"
#include "PoissonDisk.hpp"
#include "zensim/math/Vec.h"

namespace zs {

  template <typename T>
  auto sample_uniform_box(T dx, vec<int, 3> minc, vec<int, 3> maxc) {  ///< cube
    std::vector<std::array<T, 3>> data;

    for (int i = minc[0]; i < maxc[0]; ++i)
      for (int j = minc[1]; j < maxc[1]; ++j)
        for (int k = minc[2]; k < maxc[2]; ++k) {
          std::array<T, 3> cell_center;
          cell_center[0] = (i)*dx;
          cell_center[1] = (j)*dx;
          cell_center[2] = (k)*dx;

          for (int ii = -1; ii <= 1; ii = ii + 2)
            for (int jj = -1; jj <= 1; jj = jj + 2)
              for (int kk = -1; kk <= 1; kk = kk + 2) {
                std::array<T, 3> particle;
                particle[0] = cell_center[0] + ii * 0.25 * dx;
                particle[1] = cell_center[1] + jj * 0.25 * dx;
                particle[2] = cell_center[2] + kk * 0.25 * dx;
                data.push_back(particle);
              }
        }
    return data;
  }

  template <typename LS> decltype(auto) sampleFromLevelSet(const LS &ls, float dx, float ppc) {
    using T = typename LS::T;
    static constexpr int dim = LS::dim;
    auto [minCorner, maxCorner] = ls.getBoundingBox();

    PoissonDisk<T, dim> pd{};
    pd.minCorner = minCorner;
    pd.maxCorner = maxCorner;
    pd.setDistanceByPpc(dx, ppc);
    return pd.sample([&ls](const vec<T, dim> &x) { return ls.getSignedDistance(x) < 0; });
  }

}  // namespace zs
