#pragma once

#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include <zeno/zeno.h>

namespace zeno {

struct ZenoAffineMatrix : zeno::IObject {
  using mat4 = zs::vec<float, 4, 4>;
  mat4 affineMap;
};

struct ZenoParticles : zeno::IObject {
  auto &get() noexcept { return particles; }
  const auto &get() const noexcept { return particles; }
  zs::GeneralParticles particles;
  zs::ConstitutiveModelConfig model;
};

struct ZenoSparseLevelSet : zeno::IObject {
  auto &get() noexcept { return ls; }
  const auto &get() const noexcept { return ls; }
  zs::SparseLevelSet<3> ls;
};

} // namespace zeno