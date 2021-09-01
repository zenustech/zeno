#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimObject.h"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/particle/Query.hpp"
#include <zeno/NumericObject.h>

namespace zeno {

struct MakeIndexBuckets : zeno::INode {
  void apply() override {
    float radius = get_input<zeno::NumericObject>("radius")->get<float>();
    float radiusMin =
        has_input("radiusMin")
            ? get_input<zeno::NumericObject>("radiusMin")->get<float>()
            : 0.f;
    auto &particles = get_input<zeno::ZenoParticles>("ZSParticles")->get();
    ZenoIndexBuckets ibs;
    ibs.get() = zs::index_buckets_for_particles<zs::execspace_e::cuda>(
        particles, radius);

    fmt::print("done building index buckets with {} entries, {} buckets\n",
               zs::match([](auto &o) { return o.numEntries(); })(ibs.get()),
               zs::match([](auto &o) { return o.numBuckets(); })(ibs.get()));
    auto tmp = std::make_shared<zeno::ZenoIndexBuckets>(std::move(ibs));
    set_output("ZSIndexBuckets", std::move(tmp));
  }
};

ZENDEFNODE(MakeIndexBuckets, {{{"ZSParticles"},
                               {"numeric:float", "radius"},
                               {"numeric:float", "radiusMin"}},
                              {"ZSIndexBuckets"},
                              {},
                              {"GPUMPM"}});

} // namespace zeno