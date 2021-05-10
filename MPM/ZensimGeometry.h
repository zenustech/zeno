#pragma once

#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include <zen/zen.h>

namespace zenbase {

struct ZenoParticles : zen::IObject {
  auto &get() noexcept { return particles; }
  const auto &get() const noexcept { return particles; }
  zs::GeneralParticles particles;
};

struct ZenoGrid : zen::IObject {
  auto &get() noexcept { return grid; }
  const auto &get() const noexcept { return grid; }
  zs::GeneralGridBlocks grid;
};

struct ZenoSparseLevelSet : zen::IObject {
  auto &get() noexcept { return ls; }
  const auto &get() const noexcept { return ls; }
  zs::SparseLevelSet<3> ls;
};

} // namespace zenbase