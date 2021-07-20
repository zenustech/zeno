#pragma once

#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include <zeno/zeno.h>

namespace zeno {

struct ZenoParticles : zeno::IObject {
  auto &get() noexcept { return particles; }
  const auto &get() const noexcept { return particles; }
  zs::GeneralParticles particles;
  zs::ConstitutiveModelConfig model;
};

struct ZenoGrid : zeno::IObject {
  auto &get() noexcept { return grid; }
  const auto &get() const noexcept { return grid; }
  zs::GeneralGridBlocks grid;
};

struct ZenoSparseLevelSet : zeno::IObject {
  auto &get() noexcept { return ls; }
  const auto &get() const noexcept { return ls; }
  zs::SparseLevelSet<3> ls;
};

} // namespace zeno