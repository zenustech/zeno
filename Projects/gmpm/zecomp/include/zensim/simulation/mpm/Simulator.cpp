#include "Simulator.hpp"

#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Collider.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"

namespace zs {

  MPMSimulatorBuilder MPMSimulator::create() { return {}; }

  BuilderForMPMSimulator& BuilderForMPMSimulator::addScene(Scene&& scene) {
    // constitutive model
    const int particleModelOffset = this->target().particles.size();
    for (auto&& [config, geomTag, localId] : scene.models)
      if (geomTag == Scene::model_e::Particle)
        this->target().models.emplace_back(config, particleModelOffset + localId);
    // particles
    for (std::size_t i = 0; i < scene.particles.size(); ++i)
      /// range-based for loop might not be safe after moved
      this->target().particles.push_back(std::move(scene.particles[i]));
    this->target().boundaries = std::move(scene.boundaries);
#if 0
    for (auto&& boundary : this->target().boundaries) {
      match([](auto& b)
                -> std::enable_if_t<is_levelset_boundary<remove_cvref_t<decltype(b)>>::value> {
                  b = b.clone(MemoryHandle{memsrc_e::um, -5});
                },
            [](...) {})(boundary);
    }
#endif
    return *this;
  }
  BuilderForMPMSimulator& BuilderForMPMSimulator::setSimOptions(const SimOptions& ops) {
    this->target().simOptions = ops;
    return *this;
  }

  BuilderForMPMSimulator::operator MPMSimulator() noexcept {
    std::vector<MemoryHandle> memDsts(0);
    std::vector<std::vector<std::tuple<std::size_t, std::size_t>>> groups(0);
    auto searchHandle = [&memDsts](MemoryHandle mh) -> int {
      for (auto&& [id, entry] : zs::zip(zs::range(memDsts.size()), memDsts))
        if (mh.memspace() == entry.memspace() && mh.devid() == entry.devid()) return id;
      return -1;
    };
    auto searchModel = [&models = this->target().models](std::size_t objId) {
      int modelid{0};
      for (auto&& [model, id] : models) {
        if (id == objId) return modelid;
        modelid++;
      }
      return -1;
    };
    std::vector<std::size_t> numParticles(0);
    std::size_t id = 0;
    for (auto&& particles : this->target().particles) {
      match([&](auto& ps) {
        auto did = searchHandle(ps.handle());
        if (did == -1) {
          memDsts.push_back(ps.handle());
          numParticles.push_back(ps.size());
          groups.emplace_back(std::move(std::vector<std::tuple<std::size_t, std::size_t>>{
              std::make_tuple(searchModel(id), id)}));
        } else {
          numParticles[did] += ps.size();
          groups[did].push_back(std::make_tuple(searchModel(id), id));
        }
      })(particles);
      id++;
    }
    fmt::print("target processor\n");
    for (auto mh : memDsts)
      fmt::print("[{}, {}] ", magic_enum::enum_name(mh.memspace()), static_cast<int>(mh.devid()));
    fmt::print("\ntotal num particles per processor\n");
    for (auto np : numParticles) fmt::print("{} ", np);
    fmt::print("\n");
    for (auto&& [groupid, groups] : zs::zip(zs::range(groups.size()), groups)) {
      fmt::print("group {}: ", groupid);
      for (auto&& [modelid, objid] : groups) fmt::print("[{}, {}] ", modelid, objid);
      fmt::print("\n");
    }

    std::vector<std::size_t> numBlocks(numParticles.size());
    for (auto&& [dst, src] : zs::zip(numBlocks, numParticles)) dst = src / 8;
    for (auto&& [id, n] : zs::zip(range(numBlocks.size()), numBlocks))
      fmt::print("allocating {} blocks for partition {} in total!\n", n, id);

    /// particle model groups
    /// grid blocks, partitions
    this->target().gridBlocks.resize(memDsts.size());
    this->target().partitions.resize(memDsts.size());
    this->target().maxVelSqrNorms.resize(memDsts.size());
    for (auto&& [memDst, nblocks, gridBlocks, partition, maxVel] :
         zs::zip(memDsts, numBlocks, this->target().gridBlocks, this->target().partitions,
                 this->target().maxVelSqrNorms)) {
      gridBlocks = GridBlocks<GridBlock<dat32, 3, 2, 2>>{target().simOptions.dx, nblocks,
                                                         memDst.memspace(), memDst.devid()};
      partition = HashTable<i32, 3, int>{nblocks, memDst.memspace(), memDst.devid()};
      maxVel = Vector<float>{1, memDst.memspace(), memDst.devid()};
    }
    this->target().memDsts = std::move(memDsts);
    this->target().groups = std::move(groups);

    return std::move(this->target());
  }

}  // namespace zs