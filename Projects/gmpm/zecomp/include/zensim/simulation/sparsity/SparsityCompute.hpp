#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "SparsityOp.hpp"
#include "zensim/types/Iterator.h"

namespace zs {

  template <execspace_e space>
  GeneralHashTable partition_for_particles(GeneralParticles& particles, float dx, int blocklen);

}  // namespace zs