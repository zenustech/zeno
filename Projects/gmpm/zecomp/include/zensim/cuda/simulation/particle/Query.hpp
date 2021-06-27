#pragma once
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(cuda_exec_tag, const GeneralParticles &particles,
                                               float dx);

}  // namespace zs