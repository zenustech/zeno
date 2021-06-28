#include "Query.hpp"

#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/memory/MemoryResource.h"
#include "zensim/simulation/sparsity/SparsityOp.hpp"

namespace zs {

  GeneralIndexBuckets build_neighbor_list_impl(cuda_exec_tag, const GeneralParticles &particles,
                                               float dx) {
    return match([dx](const auto &pars) -> GeneralIndexBuckets {
      using particles_t = remove_cvref_t<decltype(pars)>;
      constexpr int dim = particles_t::dim;
      using indexbuckets_t = IndexBuckets<dim>;
      using vector_t = typename indexbuckets_t::vector_t;
      using table_t = typename indexbuckets_t::table_t;
      const auto memLoc = pars.space();
      const auto did = pars.devid();

      indexbuckets_t indexBuckets{};
      indexBuckets._dx = dx;
      // table
      auto &table = indexBuckets._table;
      table = {pars.size(), memLoc, did};

      auto cudaPol = cuda_exec().device(did).sync(true);
      cudaPol({table._tableSize}, CleanSparsity{exec_cuda, table});
      cudaPol({pars.size()},
              ComputeSparsity{exec_cuda, dx, 1, table, const_cast<particles_t &>(pars).X, 0});
      /// counts, offsets, indices
      // counts
      auto &counts = indexBuckets._counts;
      auto numCells = table.size() + 1;
      counts = vector_t{(std::size_t)numCells, memLoc, did};
      memset(mem_device, counts.data(), 0, sizeof(typename vector_t::value_type) * counts.size());
      auto tmp = counts;  // zero-ed array
      cudaPol({pars.size()}, SpatiallyCount{exec_cuda, dx, table, const_cast<particles_t &>(pars).X,
                                            counts, 1, 0});
      // offsets
      auto &offsets = indexBuckets._offsets;
      offsets = vector_t{(std::size_t)numCells, memLoc, did};
      exclusive_scan(cudaPol, counts.begin(), counts.end(), offsets.begin());
      // indices
      auto &indices = indexBuckets._indices;
      indices = vector_t{pars.size(), memLoc, did};
      cudaPol({pars.size()},
              SpatiallyDistribute{exec_cuda, dx, table, const_cast<particles_t &>(pars).X, tmp,
                                  offsets, indices, 1, 0});
      return indexBuckets;
    })(particles);
  }

}  // namespace zs