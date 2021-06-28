#include "Scene.hpp"

#include <filesystem>
#include <stdexcept>

#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/GeometrySampler.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/Vec.h"
#include "zensim/memory/MemoryResource.h"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/resource/Resource.h"
#include "zensim/tpls/fmt/color.h"
#include "zensim/tpls/fmt/core.h"
#include "zensim/tpls/magic_enum/magic_enum.hpp"

namespace zs {

  namespace fs = std::filesystem;

  SceneBuilder Scene::create() { return {}; }

  BuilderForSceneParticle BuilderForScene::particle() {
    return BuilderForSceneParticle{this->target()};
  }
  BuilderForSceneMesh BuilderForScene::mesh() { return BuilderForSceneMesh{this->target()}; }
  BuilderForSceneBoundary BuilderForScene::boundary() {
    return BuilderForSceneBoundary{this->target()};
  }

  BuilderForSceneParticle &BuilderForSceneParticle::addParticles(std::string fn, float dx,
                                                                 float ppc) {
    fs::path p{fn};
    ParticleModel positions{};
    if (p.extension() == ".vdb")
      positions = sample_from_vdb_file(fn, dx, ppc);
    else if (p.extension() == ".obj")
      positions = sample_from_obj_file(fn, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "does not support format {}\n", fn);
    fmt::print(fg(fmt::color::green), "done sampling {} particles [{}] with (dx: {}, ppc: {})\n",
               positions.size(), fn, dx, ppc);
    if (positions.size()) particlePositions.push_back(std::move(positions));
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::addCuboid(std::vector<float> mi,
                                                              std::vector<float> ma, float dx,
                                                              float ppc) {
    using ALS = AnalyticLevelSet<analytic_geometry_e::Cuboid, float, 3>;
    ParticleModel positions{};
    if (mi.size() == 3 && ma.size() == 3)
      positions = sampleFromLevelSet(
          ALS{vec<float, 3>{mi[0], mi[1], mi[2]}, vec<float, 3>{ma[0], ma[1], ma[2]}}, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "cuboid build config dimension error ({}, {})\n", mi.size(),
                 ma.size());
    fmt::print(
        fg(fmt::color::green),
        "done sampling {} particles [cuboid ({}, {}, {}) - ({}, {}, {})] with (dx: {}, ppc: {})\n",
        positions.size(), mi[0], mi[1], mi[2], ma[0], ma[1], ma[2], dx, ppc);
    if (positions.size()) particlePositions.push_back(std::move(positions));
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::addCube(std::vector<float> c, float len,
                                                            float dx, float ppc) {
    std::vector<float> mi{c[0] - len / 2, c[1] - len / 2, c[2] - len / 2};
    std::vector<float> ma{c[0] + len / 2, c[1] + len / 2, c[2] + len / 2};
    return addCuboid(mi, ma, dx, ppc);
  }
  BuilderForSceneParticle &BuilderForSceneParticle::addSphere(std::vector<float> c, float r,
                                                              float dx, float ppc) {
    using ALS = AnalyticLevelSet<analytic_geometry_e::Sphere, float, 3>;
    ParticleModel positions{};
    if (c.size() == 3)
      positions = sampleFromLevelSet(ALS{vec<float, 3>{c[0], c[1], c[2]}, r}, dx, ppc);
    else
      fmt::print(fg(fmt::color::red), "sphere build config dimension error center{}\n", c.size());
    fmt::print(fg(fmt::color::green),
               "done sampling {} particles [sphere ({}, {}, {}), {}] with (dx: {}, ppc: {})\n",
               positions.size(), c[0], c[1], c[2], r, dx, ppc);
    if (positions.size()) particlePositions.push_back(std::move(positions));
    return *this;
  }

  BuilderForSceneParticle &BuilderForSceneParticle::setConstitutiveModel(
      constitutive_model_e model) {
    switch (model) {
      case constitutive_model_e::EquationOfState:
        config = EquationOfStateConfig{};
        break;
      case constitutive_model_e::NeoHookean:
        config = NeoHookeanConfig{};
        break;
      case constitutive_model_e::FixedCorotated:
        config = FixedCorotatedConfig{};
        break;
      case constitutive_model_e::VonMisesFixedCorotated:
        config = VonMisesFixedCorotatedConfig{};
        break;
      case constitutive_model_e::DruckerPrager:
        config = DruckerPragerConfig{};
        break;
      case constitutive_model_e::NACC:
        config = NACCConfig{};
        break;
      default:
        fmt::print(fg(fmt::color::red), "constitutive model not known!");
        break;
    }
    return *this;
  }

  BuilderForSceneParticle &BuilderForSceneParticle::output(std::string fn) {
    displayConfig(config);
    for (auto &&[id, positions] : zip(range(particlePositions.size()), particlePositions))
      write_partio<float, 3>(fn + std::to_string(id) + ".bgeo", positions);
    return *this;
  }
  BuilderForSceneParticle &BuilderForSceneParticle::commit(MemoryHandle dst) {
    auto &scene = this->target();
    auto &dstParticles = scene.particles;
    using T = typename Particles<f32, 3>::T;
    using TV = typename Particles<f32, 3>::TV;
    using TM = typename Particles<f32, 3>::TM;
    // bridge on host
    struct {
      Vector<T> M{};
      Vector<TV> X{}, V{};
      Vector<T> J{}, logJp0{};
      Vector<TM> F{}, C{};
    } tmp;

    const bool hasPlasticity
        = config.index() == magic_enum::enum_integer(constitutive_model_e::DruckerPrager)
          || config.index() == magic_enum::enum_integer(constitutive_model_e::NACC);
    const bool hasF
        = config.index() != magic_enum::enum_integer(constitutive_model_e::EquationOfState);

    tmp.M = Vector<T>{memsrc_e::host, -1};
    tmp.X = Vector<TV>{memsrc_e::host, -1};
    tmp.V = Vector<TV>{memsrc_e::host, -1};
    if (hasF)
      tmp.F = Vector<TM>{memsrc_e::host, -1};
    else
      tmp.J = Vector<T>{memsrc_e::host, -1};
    tmp.C = Vector<TM>{memsrc_e::host, -1};
    if (hasPlasticity) tmp.logJp0 = Vector<T>{memsrc_e::host, -1};

    for (auto &positions : particlePositions) {
      tmp.M.resize(positions.size());
      tmp.X.resize(positions.size());
      tmp.V.resize(positions.size());
      if (hasF)
        tmp.F.resize(positions.size());
      else
        tmp.J.resize(positions.size());
      tmp.C.resize(positions.size());
      if (hasPlasticity) tmp.logJp0.resize(positions.size());
      /// -> bridge
      // default mass, vel, F
      assert_with_msg(sizeof(float) * 3 == sizeof(TV), "fatal: TV size not as expected!");
      {
        std::vector<T> defaultMass(positions.size(), match([](auto &config) {
                                     return config.rho * config.volume;
                                   })(config));
        memcpy(tmp.M.head(), defaultMass.data(), sizeof(T) * positions.size());

        memcpy(tmp.X.head(), positions.data(), sizeof(TV) * positions.size());

        std::vector<std::array<T, 3>> defaultVel(positions.size(), {0, 0, 0});
        memcpy(tmp.V.head(), defaultVel.data(), sizeof(TV) * positions.size());

        if (hasF) {
          std::vector<std::array<T, 3 * 3>> defaultF(positions.size(), {1, 0, 0, 0, 1, 0, 0, 0, 1});
          memcpy(tmp.F.head(), defaultF.data(), sizeof(TM) * positions.size());
        } else {
          std::vector<T> defaultJ(positions.size(), 1.f);
          memcpy(tmp.J.head(), defaultJ.data(), sizeof(T) * positions.size());
        }
        std::vector<std::array<T, 3 * 3>> defaultC(positions.size(), {0, 0, 0, 0, 0, 0, 0, 0, 0});
        memcpy(tmp.C.head(), defaultC.data(), sizeof(TM) * positions.size());
        if (hasPlasticity) {
          std::vector<T> defaultLogJp0(
              positions.size(),
              match([](auto &config) -> decltype(config.logJp0) { return config.logJp0; },
                    [](...) { return 0.f; })(config));
          memcpy(tmp.logJp0.head(), defaultLogJp0.data(), sizeof(T) * positions.size());
        }
      }
      // constitutive model
      scene.models.emplace_back(config, Scene::model_e::Particle, dstParticles.size());
      // particles
      dstParticles.push_back(Particles<f32, 3>{});
      match(
          [&tmp, &dst, hasF, hasPlasticity, this](Particles<f32, 3> &pars) {
            pars.M = tmp.M.clone(dst);
            pars.X = tmp.X.clone(dst);
            pars.V = tmp.V.clone(dst);
            if (hasF)
              pars.F = tmp.F.clone(dst);
            else
              pars.J = tmp.J.clone(dst);
            pars.C = tmp.C.clone(dst);
            if (hasPlasticity) pars.logJp = tmp.logJp0.clone(dst);
            fmt::print("moving {} paticles [{}, {}]\n", pars.X.size(),
                       magic_enum::enum_name(pars.X.memspace()), static_cast<int>(pars.X.devid()));
          },
          [](...) {})(dstParticles.back());
    }
    particlePositions.clear();
    return *this;
  }

  ///
  BuilderForSceneBoundary &BuilderForSceneBoundary::addSparseLevelSet(std::string fn) {
    auto vdbGrid = zs::loadFloatGridFromVdbFile(fn);
    sparseLevelSets.emplace_back(zs::convertFloatGridToSparseLevelSet(vdbGrid));
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::addVdbLevelset(std::string fn, float dx) {
    // ;
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::addCuboid(std::vector<float> mi,
                                                              std::vector<float> ma) {
    cuboids.emplace_back(vec<float, 3>{mi[0], mi[1], mi[2]}, vec<float, 3>{ma[0], ma[1], ma[2]});
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::addCube(std::vector<float> c, float len) {
    vec<float, 3> o{c[0], c[1], c[2]};
    cuboids.emplace_back(o - len / 2, o + len / 2);
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::addSphere(std::vector<float> c, float r) {
    spheres.emplace_back(vec<float, 3>{c[0], c[1], c[2]}, r);
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::addPlane(std::vector<float> o,
                                                             std::vector<float> n) {
    planes.emplace_back(vec<float, 3>{o[0], o[1], o[2]}, vec<float, 3>{n[0], n[1], n[2]});
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::setBoundaryType(collider_e type) {
    boundaryType = type;
    return *this;
  }
  BuilderForSceneBoundary &BuilderForSceneBoundary::commit(MemoryHandle dst) {
    auto &scene = this->target();
    auto &dstBoundaries = scene.boundaries;
    /// cuboid
    {
      for (auto &&als : cuboids) dstBoundaries.emplace_back(Collider{als, boundaryType});
      cuboids.clear();
    }
    /// sphere
    {
      for (auto &&als : spheres) dstBoundaries.emplace_back(Collider{als, boundaryType});
      spheres.clear();
    }
    /// plane
    {
      for (auto &&als : planes) dstBoundaries.emplace_back(Collider{als, boundaryType});
      planes.clear();
    }
    /// vdb levelset
    {
      // for (int i = 0; i < sparseLevelSets.size(); ++i)
      for (auto &&spls : sparseLevelSets)
        dstBoundaries.emplace_back(LevelSetBoundary{spls.clone(dst), boundaryType});
      sparseLevelSets.clear();
    }
    return *this;
  }

  ///
  SimOptionsBuilder SimOptions::create() { return {}; }
}  // namespace zs