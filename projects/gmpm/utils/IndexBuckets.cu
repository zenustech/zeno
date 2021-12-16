#include "../mpm/Structures.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/NumericObject.h>

namespace zeno {

struct MakeZSBuckets : zeno::INode {
  void apply() override {
    float radius = get_input<NumericObject>("radius")->get<float>();
    float radiusMin = has_input("radiusMin")
                          ? get_input<NumericObject>("radiusMin")->get<float>()
                          : 0.f;
    auto &pars = get_input<ZenoParticles>("ZSParticles")->getParticles();

    auto out = std::make_shared<ZenoIndexBuckets>();
    auto &ibs = out->get();

    using namespace zs;
    auto allocator = pars.get_allocator();
    auto mloc = allocator.location;
    ibs._dx = radius + radius;
    /// table
    auto &partition = ibs._table;
    using Partition = RM_CVREF_T(partition);
    partition = Partition{pars.size(), pars.memspace(), pars.devid()};

    // clean
    auto cudaPol = cuda_exec().device(0);
    cudaPol(range(partition._tableSize),
            [table = proxy<execspace_e::cuda>(partition)] __device__(
                size_t i) mutable {
              table._table.keys[i] =
                  Partition::key_t::uniform(Partition::key_scalar_sentinel_v);
              table._table.indices[i] = Partition::sentinel_v;
              table._table.status[i] = -1;
              if (i == 0)
                *table._cnt = 0;
            });
    // compute sparsity
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 ibs = proxy<execspace_e::cuda>(
                                     ibs)] __device__(size_t pi) mutable {
      auto x = pars.template pack<3>("pos", pi);
      auto coord = ibs.bucketCoord(x);
      ibs.table.insert(coord);
    });
    auto numCells = partition.size() + 1;

    /// counts
    using index_type = typename ZenoIndexBuckets::buckets_t::index_type;
    auto &counts = ibs._counts;
    counts = counts.clone(mloc);
    counts.resize(numCells);
    zs::memset(mem_device, counts.data(), 0, sizeof(index_type) * numCells);
#if 0
    cudaPol(range(counts.size()),
            [counts = proxy<execspace_e::cuda>(counts)] __device__(
                size_t i) mutable { counts[i] = 0; });
#endif
    auto tmp = counts; // for index distribution later
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 ibs = proxy<execspace_e::cuda>(
                                     ibs)] __device__(size_t pi) mutable {
      auto pos = pars.template pack<3>("pos", pi);
      auto coord = ibs.bucketCoord(pos);
      atomic_add(exec_cuda, (index_type *)&ibs.counts[ibs.table.query(coord)],
                 (index_type)1);
    });
    /// offsets
    auto &offsets = ibs._offsets;
    offsets = offsets.clone(mloc);
    offsets.resize(numCells);
    exclusive_scan(cudaPol, std::begin(counts), std::end(counts),
                   std::begin(offsets));
    /// indices
    auto &indices = ibs._indices;
    indices = indices.clone(mloc);
    indices.resize(pars.size());
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 counts = proxy<execspace_e::cuda>(tmp),
                                 ibs = proxy<execspace_e::cuda>(
                                     ibs)] __device__(size_t pi) mutable {
      auto pos = pars.template pack<3>("pos", pi);
      auto coord = ibs.bucketCoord(pos);
      auto cellno = ibs.table.query(coord);
      auto localno =
          atomic_add(exec_cuda, (index_type *)&counts[cellno], (index_type)1);
      ibs.indices[ibs.offsets[cellno] + localno] = (index_type)pi;
    });

    fmt::print("done building index buckets with {} entries, {} buckets\n",
               ibs.numEntries(), ibs.numBuckets());

    set_output("MakeZSBuckets", std::move(out));
  }
};

ZENDEFNODE(MakeZSBuckets, {{{"ZSParticles"},
                            {"numeric:float", "radius"},
                            {"numeric:float", "radiusMin"}},
                           {"ZSIndexBuckets"},
                           {},
                           {"MPM"}});

} // namespace zeno