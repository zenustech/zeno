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
#include <zeno/NumericObject.h>
namespace zeno {

struct SpatialPartitionForParticles : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing SpatialPartitionForParticles\n");
    // auto partition = zeno::IObject::make<ZenoPartition>();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();

    auto dx = std::get<float>(get_param("dx"));
    if (has_input("Dx")) {
      dx = get_input("Dx")->as<NumericObject>()->get<float>();
    }
    auto blocklen = std::get<int>(get_param("block_side_length"));
    // auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    ZenoParticleObjects parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ZenoParticleList>("ZSParticles")) {
      auto &list = get_input<ZenoParticleList>("ZSParticles")->get();
      parObjPtrs.insert(parObjPtrs.end(), list.begin(), list.end());
    } else if (has_input<ListObject>("ZSParticles")) {
      // auto &objSharedPtrLists = *get_input("ZSParticles")->as<ListObject>();
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }

    std::size_t cnt = 0;
    zs::MemoryLocation mloc;
    for (auto &&parObjPtr : parObjPtrs) {
      cnt += zs::match([](auto &p) { return p.size(); })(parObjPtr->get());
      mloc = zs::match([](auto &p) { return p.memoryLocation(); })(
          parObjPtr->get());
    }
    using TableT = zs::HashTable<zs::i32, 3, int>;
    // zs::HashTable<zs::i32, 3, int> ret{cnt, mh.memspace(), mh.devid()};
    TableT &ret = std::get<TableT>(partition);
    if (ret._tableSize < ret.evaluateTableSize(cnt))
      ret = TableT{cnt, mloc.memspace(), mloc.devid()};

    auto cudaPol = zs::cuda_exec().device(0);
    cudaPol({ret._tableSize},
            zs::CleanSparsity{zs::wrapv<zs::execspace_e::cuda>{}, ret});
    for (auto &&parObjPtr : parObjPtrs) {
      auto &particles = parObjPtr->get();
      zs::match([&ret, &cudaPol, dx, blocklen](auto &p) {
        cudaPol({p.size()},
                zs::ComputeSparsity{zs::wrapv<zs::execspace_e::cuda>{}, dx,
                                    blocklen, ret, p.attrVector("pos")});
      })(particles);
    }

    // partition->get() = std::move(ret);
    fmt::print(fg(fmt::color::cyan),
               "done executing SpatialPartitionForParticles\n");
    // set_output("ZSPartition", partition);
  }
};

static int defSpatialPartitionForParticles = zeno::defNodeClass<
    SpatialPartitionForParticles>(
    "SpatialPartitionForParticles",
    {/* inputs: */ {"ZSPartition", "ZSParticles", "Dx"},
     /* outputs: */ {},
     /* params: */ {{"float", "dx", "1"}, {"int", "block_side_length", "4"}},
     /* category: */ {"GPUMPM"}});

} // namespace zeno