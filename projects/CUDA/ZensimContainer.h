#pragma once

#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include <zeno/zeno.h>

namespace zeno {

struct ZenoPartition : zeno::IObject {
  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }
  zs::GeneralHashTable table;
};

struct ZenoIndexBuckets : zeno::IObject {
  auto &get() noexcept { return ibs; }
  const auto &get() const noexcept { return ibs; }
  zs::GeneralIndexBuckets ibs;
};

} // namespace zeno