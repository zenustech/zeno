#include "../ZensimContainer.h"
#include "../ZensimGeometry.h"
#include "../ZensimModel.h"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/cuda/simulation/transfer/G2P.hpp"
#include "zensim/cuda/simulation/transfer/P2G.hpp"
#include "zensim/cuda/container/HashTable.hpp"

namespace zenbase {

struct P2G : zen::INode {
  void apply() override {
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();

    auto stepDt = std::get<float>(get_param("dt"));

    auto cudaPol = zs::cuda_exec().device(0);
    zs::match([&](auto &constitutiveModel, auto &obj, auto &partition,
                  auto &grid)
                  -> std::enable_if_t<
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(partition)>::dim &&
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(grid)>::dim> {
      cudaPol({obj.size()},
              zs::P2GTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                              zs::wrapv<zs::transfer_scheme_e::apic>{}, stepDt,
                              constitutiveModel, obj, partition, grid});
    })(model, particles, partition, grid);
  }
};

static int defP2G = zen::defNodeClass<P2G>(
    "P2G", {/* inputs: */ {"ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
            /* outputs: */ {},
            /* params: */ {{"float", "dt", "0.0001"}},
            /* category: */ {"simulation"}});

struct G2P : zen::INode {
  void apply() override {
    auto &model = get_input("ZSModel")->as<ZenoConstitutiveModel>()->get();
    auto &grid = get_input("ZSGrid")->as<ZenoGrid>()->get();
    auto &partition = get_input("ZSPartition")->as<ZenoPartition>()->get();
    auto &particles = get_input("ZSParticles")->as<ZenoParticles>()->get();

    auto stepDt = std::get<float>(get_param("dt"));

    auto cudaPol = zs::cuda_exec().device(0);
    zs::match([&](auto &constitutiveModel, auto &grid, auto &partition,
                  auto &obj)
                  -> std::enable_if_t<
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(partition)>::dim &&
                      zs::remove_cvref_t<decltype(obj)>::dim ==
                          zs::remove_cvref_t<decltype(grid)>::dim> {
      cudaPol({obj.size()},
              zs::G2PTransfer{zs::wrapv<zs::execspace_e::cuda>{},
                              zs::wrapv<zs::transfer_scheme_e::apic>{}, stepDt,
                              constitutiveModel, grid, partition, obj});
    })(model, grid, partition, particles);
  }
};

static int defG2P = zen::defNodeClass<G2P>(
    "G2P", {/* inputs: */ {"ZSModel", "ZSParticles", "ZSGrid", "ZSPartition"},
            /* outputs: */ {},
            /* params: */ {{"float", "dt", "0.0001"}},
            /* category: */ {"simulation"}});

} // namespace zenbase