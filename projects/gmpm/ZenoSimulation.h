#pragma once
#include "zensim/geometry/Collider.h"
#include <zeno/zeno.h>

namespace zen {

struct ZenoBoundary : zen::IObject {
  auto &get() noexcept { return boundary; }
  const auto &get() const noexcept { return boundary; }
  zs::GeneralBoundary boundary;
};

} // namespace zen