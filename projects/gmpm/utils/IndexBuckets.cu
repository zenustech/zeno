#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
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
    auto cudaPol = cuda_exec().device(0);
    spatial_hashing(cudaPol, pars, radius + radius, ibs);

    fmt::print("done building index buckets with {} entries, {} buckets\n",
               ibs.numEntries(), ibs.numBuckets());

    set_output("ZSIndexBuckets", std::move(out));
  }
};

ZENDEFNODE(MakeZSBuckets, {{{"ZSParticles"},
                            {"numeric:float", "radius"},
                            {"numeric:float", "radiusMin"}},
                           {"ZSIndexBuckets"},
                           {},
                           {"MPM"}});

} // namespace zeno