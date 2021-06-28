#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimObject.h"
//#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/simulation/sparsity/SparsityCompute.hpp"
#include "zensim/simulation/sparsity/SparsityOp.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"

namespace zen {

struct SpatialPartitionForParticles : zen::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing SpatialPartitionForParticles\n");
    auto partition = zen::IObject::make<ZenoPartition>();

    auto dx = std::get<float>(get_param("dx"));
    auto blocklen = std::get<int>(get_param("block_side_length"));
    // auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    ZenoParticleObjects parObjPtrs{};
    if (get_input("ZSParticles")->as<ZenoParticles>())
      parObjPtrs.push_back(get_input("ZSParticles")->as<ZenoParticles>());
    else if (get_input("ZSParticles")->as<ZenoParticleList>()) {
      auto &list = get_input("ZSParticles")->as<ZenoParticleList>()->get();
      parObjPtrs.insert(parObjPtrs.end(), list.begin(), list.end());
    }

    std::size_t cnt = 0;
    zs::MemoryHandle mh;
    for (auto &&parObjPtr : parObjPtrs) {
      cnt += zs::match([](auto &p) { return p.size(); })(parObjPtr->get());
      mh = zs::match([](auto &p) { return p.handle(); })(parObjPtr->get());
    }
    zs::HashTable<zs::i32, 3, int> ret{cnt, mh.memspace(), mh.devid()};

    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol({ret._tableSize},
            zs::CleanSparsity{zs::wrapv<zs::execspace_e::cuda>{}, ret});
    for (auto &&parObjPtr : parObjPtrs) {
      auto &particles = parObjPtr->get();
      zs::match([&ret, &cudaPol, dx, blocklen](auto &p) {
        cudaPol({p.size()},
                zs::ComputeSparsity{zs::wrapv<zs::execspace_e::cuda>{}, dx,
                                    blocklen, ret, p.X});
      })(particles);
    }
    partition->get() = std::move(ret);

    fmt::print(fg(fmt::color::cyan),
               "done executing SpatialPartitionForParticles\n");
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

} // namespace zen