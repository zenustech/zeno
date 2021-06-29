#pragma once

#include "zensim/container/HashTable.hpp"
#include <zeno/zeno.h>

namespace zen {

struct ZenoPartition : zen::IObject {
  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }
  zs::GeneralHashTable table;
};


} // namespace zen