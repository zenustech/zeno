#include "../Utils.hpp"
#include "Structures.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/types/DictObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ComputeParticleVolume : INode {
  void apply() override {
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    auto buckets = std::make_shared<ZenoIndexBuckets>();
    auto &ibs = buckets->get();

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    bool first = true;

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      spatial_hashing(cudaPol, pars, grid.dx, ibs, first, true);
      first = false;
    }

    for (auto &&parObjPtr : parObjPtrs) {
      auto &pars = parObjPtr->getParticles();
      cudaPol(range(pars.size()),
              [pars = proxy<execspace_e::cuda>({}, pars),
               ibs = proxy<execspace_e::cuda>(ibs),
               density = parObjPtr->getModel().density,
               cellVol =
                   grid.dx * grid.dx * grid.dx] __device__(size_t pi) mutable {
                auto pos = pars.template pack<3>("pos", pi);
                auto coord = ibs.bucketCoord(pos);
                const auto bucketNo = ibs.table.query(coord);
                // bucketNo should be > 0
                const auto cnt = ibs.counts[bucketNo];
                pars("vol", pi) = cellVol / cnt;
              });
    }
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ComputeParticleVolume,
           {
               {{"ZenoParticles", "ZSParticles"}, {"ZenoGrid", "ZSGrid"}},
               {{"ZenoParticles", "ZSParticles"}},
               {},
               {"MPM"},
           });

} // namespace zeno
