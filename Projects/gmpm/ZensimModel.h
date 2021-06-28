#pragma once

#include "zensim/physics/ConstitutiveModel.hpp"
#include <zen/zen.h>

namespace zen {

struct ZenoConstitutiveModel : zen::IObject {
  auto &get() noexcept { return model; }
  const auto &get() const noexcept { return model; }
  zs::ConstitutiveModelConfig model;
};

} // namespace zen