#pragma once

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

} // namespace zenbase