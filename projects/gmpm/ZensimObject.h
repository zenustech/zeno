#pragma once

#include "ZenoSimulation.h"
#include "ZensimContainer.h"
#include "ZensimGeometry.h"
#include "ZensimModel.h"

#include <zeno/zeno.h>

namespace zen {

using ZenoConstitutiveModels = std::vector<ZenoConstitutiveModel *>;
using ZenoParticleObjects = std::vector<ZenoParticles *>;
using ZenoBoundaries = std::vector<ZenoBoundary *>;

struct ZenoParticleList : zen::IObject {
  auto &get() noexcept { return objects; }
  const auto &get() const noexcept { return objects; }
  ZenoParticleObjects objects;
};

struct ZenoZensimObjects : zen::IObject {
  auto &get() noexcept { return objects; }
  const auto &get() const noexcept { return objects; }
  zs::variant<ZenoConstitutiveModels, ZenoParticleObjects, ZenoBoundaries>
      objects;
};

} // namespace zen