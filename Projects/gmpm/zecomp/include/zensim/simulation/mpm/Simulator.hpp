#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/memory/MemoryResource.h"
#include "zensim/resource/Resource.h"
#include "zensim/simulation/init/Scene.hpp"
#include "zensim/types/BuilderBase.hpp"

namespace zs {

  struct MPMSimulatorBuilder;
  struct BuilderForMPMSimulator;
  struct MPMSimulator {
    /// construct
    static MPMSimulatorBuilder create();

    std::size_t numModels() const noexcept { return particles.size(); }
    std::size_t numPartitions() const noexcept { return partitions.size(); }
    float getMaxVel(int partI) const {
      Vector<float> ret{1, memsrc_e::host, -1};
      copy(MemoryEntity{ret.base(), ret.data()},
           MemoryEntity{maxVelSqrNorms[partI].base(),
                        const_cast<float*>(maxVelSqrNorms[partI].data())},
           sizeof(float));
      return ret[0];
    }
    float* maxVelPtr(int partI) { return maxVelSqrNorms[partI].data(); }
    const float* maxVelPtr(int partI) const { return maxVelSqrNorms[partI].data(); }

    /// particle
    std::vector<GeneralParticles> particles;
    std::vector<std::tuple<ConstitutiveModelConfig, std::size_t>>
        models;  // (constitutive model, id)
    /// parallel execution helper
    std::vector<MemoryHandle> memDsts;
    std::vector<std::vector<std::tuple<std::size_t, std::size_t>>> groups;  // (model id, object id)
    /// background grid
    std::vector<GeneralGridBlocks> gridBlocks;
    /// sparsity info (hash table)
    std::vector<GeneralHashTable> partitions;
    std::vector<Vector<float>> maxVelSqrNorms;
    /// transfer operator
    /// boundary
    std::vector<GeneralBoundary> boundaries;
    // apic/ flip
    /// simulation setup
    SimOptions simOptions;
  };

  template <execspace_e, typename Simulator, typename = void> struct MPMSimulatorProxy;

  template <execspace_e ExecSpace> constexpr decltype(auto) proxy(MPMSimulator& simulator) {
    return MPMSimulatorProxy<ExecSpace, MPMSimulator>{simulator};
  }

  struct BuilderForMPMSimulator : BuilderFor<MPMSimulator> {
    explicit BuilderForMPMSimulator(MPMSimulator& simulator)
        : BuilderFor<MPMSimulator>{simulator} {}

    BuilderForMPMSimulator& addScene(Scene&& scene);
    BuilderForMPMSimulator& setSimOptions(const SimOptions& ops);

    operator MPMSimulator() noexcept;
  };

  struct MPMSimulatorBuilder : BuilderForMPMSimulator {
    MPMSimulatorBuilder() : BuilderForMPMSimulator{_simulator} {}

  protected:
    MPMSimulator _simulator;
  };

}  // namespace zs