#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
//#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/simulation/sparsity/SparsityOp.hpp"
#include <zen/VDBGrid.h>

namespace zenbase {

struct SpatialPartitionForParticles : zen::INode {
  void apply() override {
    auto partition = zen::IObject::make<ZenoPartition>();

    auto dx = std::get<float>(get_param("dx"));
    auto blocklen = std::get<int>(get_param("block_side_length"));
    // pass in FloatGrid::Ptr
    auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();

#if 0
    zs::HashTable<int, 3, zs::i64> ret{zs::match([](auto &particles) {
                                         return particles.size() / blocklen /
                                                blocklen / blocklen;
                                       })(particles),
                                       zs::memsrc_e::um, 0};
    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol({ret._tableSize},
            zs::CleanSparsity{zs::wrapv<zs::execspace_e::cuda>{}, ret});

    zs::match([&ret, dx, blocklen](auto &particles) {
      cudaPol({particles.size()},
              zs::ComputeSparsity{zs::wrapv<zs::execspace_e::cuda>{}, dx,
                                  blocklen, ret, particles.X});
    })(particles);

    partition->get() = std::move(ret);
#endif
    set_output("ZSPartition", partition);
  }
};

static int defSpatialPartitionForParticles = zen::defNodeClass<
    SpatialPartitionForParticles>(
    "SpatialPartitionForParticles",
    {/* inputs: */ {"ZSParticles"},
     /* outputs: */ {"ZSPartition"},
     /* params: */ {{"float", "dx", "1"}, {"int", "block_side_length", "4"}},
     /* category: */ {"simulation"}});

} // namespace zenbase