#pragma once

#include "ZensimContainer.h"
#include "ZensimGeometry.h"
#include "ZensimModel.h"

#include <zen/zen.h>

namespace zenbase {

using ZenoConstitutiveModels = std::vector<ZenoConstitutiveModel>;
using ZenoParticleObjects = std::vector<ZenoParticles>;

struct ZenoZensimObjects : zen::IObject {
  auto &get() noexcept { return objects; }
  const auto &get() const noexcept { return objects; }
  zs::variant<ZenoConstitutiveModels, ZenoParticleObjects> objects;
};

} // namespace zenbase