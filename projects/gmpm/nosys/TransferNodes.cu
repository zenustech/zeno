#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimModel.h"
#include "../ZensimObject.h"
#include "zensim/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/simulation/transfer/G2P.hpp"
#include "zensim/simulation/transfer/P2G.hpp"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/format.h"
#include <zeno/NumericObject.h>

namespace zeno {

struct P2G : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing P2G\n");
    // deprecated
    // auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    // auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    ZenoParticleObjects parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ZenoParticleList>("ZSParticles")) {
      auto &list = get_input<ZenoParticleList>("ZSParticles")->get();
      parObjPtrs.insert(parObjPtrs.end(), list.begin(), list.end());
    } else if (has_input<ListObject>("ZSParticles")) {
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto &grid = get_input<ZenoGrid>("ZSGrid")->get();

    // auto stepDt = std::get<float>(get_param("dt"));
    auto stepDt = get_input("dt")->as<zeno::NumericObject>()->get<float>();

    auto cudaPol = zs::cuda_exec().device(0);

    for (auto &&parObjPtr : parObjPtrs) {
      auto &particles = parObjPtr->get();
      zs::match(
          [&](const auto &constitutiveModel, auto &obj, auto &partition,
              auto &grid)
              -> std::enable_if_t<
                  zs::remove_cvref_t<decltype(obj)>::dim ==
                      zs::remove_cvref_t<decltype(partition)>::dim &&
                  zs::remove_cvref_t<decltype(obj)>::dim ==
                      zs::remove_cvref_t<decltype(grid)>::dim> {
            cudaPol({obj.size()},
                    zs::P2GTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                                    zs::wrapv<zs::transfer_scheme_e::apic>{},
                                    stepDt, constitutiveModel, obj, partition,
                                    grid});
          },
          [](...) { throw std::runtime_error("not implemented!"); })(
          parObjPtr->model, particles, partition, grid);
    }
    fmt::print(fg(fmt::color::cyan), "done executing P2G\n");
  }
};

static int defP2G = zeno::defNodeClass<P2G>(
    "P2G",
    {/* inputs: */ {"dt", "ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
     /* outputs: */ {},
     /* params: */ {/*{"float", "dt", "0.0001"}*/},
     /* category: */ {"simulation"}});

struct G2P : zeno::INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing G2P\n");
    // deprecated
    // auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    // auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    ZenoParticleObjects parObjPtrs{};
    if (has_input<ZenoParticles>("ZSParticles"))
      parObjPtrs.push_back(get_input<ZenoParticles>("ZSParticles").get());
    else if (has_input<ZenoParticleList>("ZSParticles")) {
      auto &list = get_input<ZenoParticleList>("ZSParticles")->get();
      parObjPtrs.insert(parObjPtrs.end(), list.begin(), list.end());
    } else if (has_input<ListObject>("ZSParticles")) {
      auto &objSharedPtrLists = *get_input<ListObject>("ZSParticles");
      for (auto &&objSharedPtr : objSharedPtrLists.get())
        if (auto ptr = dynamic_cast<ZenoParticles *>(objSharedPtr.get());
            ptr != nullptr)
          parObjPtrs.push_back(ptr);
    }

    // auto stepDt = std::get<float>(get_param("dt"));
    auto stepDt = get_input("dt")->as<zeno::NumericObject>()->get<float>();

    auto cudaPol = zs::cuda_exec().device(0);
    for (auto &&parObjPtr : parObjPtrs) {
      auto &particles = parObjPtr->get();
      zs::match(
          [&](const auto &constitutiveModel, auto &grid, auto &partition,
              auto &obj)
              -> std::enable_if_t<
                  zs::remove_cvref_t<decltype(obj)>::dim ==
                      zs::remove_cvref_t<decltype(partition)>::dim &&
                  zs::remove_cvref_t<decltype(obj)>::dim ==
                      zs::remove_cvref_t<decltype(grid)>::dim> {
            // fmt::print("{} particles g2p\n", obj.size());
            cudaPol({obj.size()},
                    zs::G2PTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                                    zs::wrapv<zs::transfer_scheme_e::apic>{},
                                    stepDt, constitutiveModel, grid, partition,
                                    obj});
          },
          [](...) { throw std::runtime_error("not implemented!"); })(
          parObjPtr->model, grid, partition, particles);
    }
    fmt::print(fg(fmt::color::cyan), "done executing G2P\n");
  }
};

static int defG2P = zeno::defNodeClass<G2P>(
    "G2P",
    {/* inputs: */ {"dt", "ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
     /* outputs: */ {},
     /* params: */ {/*{"float", "dt", "0.0001"}*/},
     /* category: */ {"simulation"}});

} // namespace zeno