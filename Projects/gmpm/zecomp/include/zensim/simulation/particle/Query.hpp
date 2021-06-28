#pragma once
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Structurefree.hpp"
#if ZS_ENABLE_CUDA
#  include "zensim/cuda/simulation/particle/Query.hpp"
#endif

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(host_exec_tag, const GeneralParticles &particles,
                                               float dx);

  template <typename ExecTag>
  auto build_neighbor_list(ExecTag tag, const GeneralParticles &particles, float dx)
      -> remove_cvref_t<decltype(build_neighbor_list_impl(tag, particles, dx),
                                 std::declval<GeneralIndexBuckets>())> {
    return build_neighbor_list_impl(tag, particles, dx);
  }

  template <typename ExecTag, typename... Args>
  GeneralIndexBuckets build_neighbor_list(ExecTag, Args &&...) {
    throw std::runtime_error(fmt::format("build_neighbor_list(tag {}, ...) not implemented\n",
                                         get_execution_space_tag(ExecTag{})));
    return {};
  }

  GeneralIndexBuckets build_neighbor_list(const GeneralParticles &particles, float dx);

}  // namespace zs