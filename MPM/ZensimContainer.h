#pragma once

#include "zensim/container/Hashtable.hpp"
#include <zen/zen.h>

namespace zenbase {

struct ZenoPartition : zen::IObject {
  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }
  zs::GeneralHashTable table;
};


} // namespace zenbase