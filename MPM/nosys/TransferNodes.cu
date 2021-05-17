#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimModel.h"
#include "zensim/cuda/container/HashTable.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/transfer/G2P.hpp"
#include "zensim/cuda/simulation/transfer/P2G.hpp"
#include <zen/NumericObject.h>

namespace zenbase {

struct P2G : zen::INode {
  void apply() override {
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();

    // auto stepDt = std::get<float>(get_param("dt"));
    auto stepDt = get_input("dt")->as<zenbase::NumericObject>()->get<float>();

    auto cudaPol = zs::cuda_exec().device(0);
#if 0
    zs::match([](auto &m) { fmt::print("model type: {}\n", zs::demangle(m)); })(
        model);
    zs::match([](auto &p) {
      fmt::print("particle type: {}\n", zs::demangle(p));
    })(particles);
    zs::match([](auto &t) { fmt::print("table type: {}\n", zs::demangle(t)); })(
        partition);
    zs::match([](auto &g) { fmt::print("grid type: {}\n", zs::demangle(g)); })(
        grid);
#elif 0
    fmt::print("start displaying\n");
    fmt::print("par {}\n", particles.index());
    fmt::print("table {}\n", partition.index());
    fmt::print("grid {}\n", grid.index());
    fmt::print("model {}\n", model.index());
#endif
    zs::match([&](auto &constitutiveModel, auto &obj, auto &partition,
                  auto &grid)
                  -> std::enable_if_t<
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(partition)>::dim &&
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(grid)>::dim> {
#if 0
      fmt::print("par addr: {}\n", obj.X.self().address());
      fmt::print("table addr: {}\n", partition.self().address());
      fmt::print("grid addr: {}\n", grid.blocks.self().address());
#endif
      cudaPol({obj.size()},
              zs::P2GTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                              zs::wrapv<zs::transfer_scheme_e::apic>{}, stepDt,
                              constitutiveModel, obj, partition, grid});
    })(model, particles, partition, grid);
  }
};

static int defP2G = zen::defNodeClass<P2G>(
    "P2G",
    {/* inputs: */ {"dt", "ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
     /* outputs: */ {},
     /* params: */ {/*{"float", "dt", "0.0001"}*/},
     /* category: */ {"simulation"}});

struct G2P : zen::INode {
  void apply() override {
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();

    // auto stepDt = std::get<float>(get_param("dt"));
    auto stepDt = get_input("dt")->as<zenbase::NumericObject>()->get<float>();

    auto cudaPol = zs::cuda_exec().device(0);
    zs::match([&](auto &constitutiveModel, auto &grid, auto &partition,
                  auto &obj)
                  -> std::enable_if_t<
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(partition)>::dim &&
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(grid)>::dim> {
      // fmt::print("{} particles g2p\n", obj.size());
      cudaPol({obj.size()},
              zs::G2PTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                              zs::wrapv<zs::transfer_scheme_e::apic>{}, stepDt,
                              constitutiveModel, grid, partition, obj});
    })(model, grid, partition, particles);
  }
};

static int defG2P = zen::defNodeClass<G2P>(
    "G2P",
    {/* inputs: */ {"dt", "ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
     /* outputs: */ {},
     /* params: */ {/*{"float", "dt", "0.0001"}*/},
     /* category: */ {"simulation"}});

} // namespace zenbase