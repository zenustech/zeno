#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/physics/constitutive_models/EquationOfState.hpp"
#include "zensim/physics/constitutive_models/FixedCorotated.h"
#include "zensim/physics/constitutive_models/NeoHookean.hpp"
#include "zensim/physics/constitutive_models/StvkWithHencky.hpp"
#include "zensim/physics/plasticity_models/VonMisesCapped.hpp"
#include "zensim/resource/Resource.h"
#include <zeno/zeno.h>

namespace zeno {

using ElasticModel =
    zs::variant<zs::FixedCorotated<float>, zs::NeoHookean<float>,
                zs::StvkWithHencky<float>>;
using PlasticModel = zs::variant<std::monostate>;

struct ZenoConstitutiveModel : IObject {
  enum elastic_model_e { Fcr, Nhk, Stvk };
  enum plastic_model_e { None, VonMises, DruckerPrager, Camclay };

  auto &getElasticModel() noexcept { return elasticModel; }
  const auto &getElasticModel() const noexcept { return elasticModel; }
  auto &getPlasticModel() noexcept { return plasticModel; }
  const auto &getPlasticModel() const noexcept { return plasticModel; }

  template <elastic_model_e I> auto &getElasticModel() noexcept {
    return std::get<I>(elasticModel);
  }
  template <elastic_model_e I> const auto &getElasticModel() const noexcept {
    return std::get<I>(elasticModel);
  }
  template <plastic_model_e I> auto &getPlasticModel() noexcept {
    return std::get<I>(plasticModel);
  }
  template <plastic_model_e I> const auto &getPlasticModel() const noexcept {
    return std::get<I>(plasticModel);
  }

  bool hasF() const noexcept { return elasticModel.index() < 3; }
  bool hasPlasticity() const noexcept { return plasticModel.index() != 0; }

  float volume, density;
  ElasticModel elasticModel;
  PlasticModel plasticModel;
};

struct ZenoParticles : IObject {
  using particles_t =
      zs::TileVector<float, 32, unsigned char, zs::ZSPmrAllocator<false>>;
  auto &getParticles() noexcept { return particles; }
  const auto &getParticles() const noexcept { return particles; }
  auto &getModel() noexcept { return model; }
  const auto &getModel() const noexcept { return model; }
  particles_t particles{};
  ZenoConstitutiveModel model{};
};

struct ZenoGrid : IObject {
  using grid_t =
      zs::Grid<float, 3, 4, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return grid; }
  const auto &get() const noexcept { return grid; }
  grid_t grid;
};

struct ZenoPartition : IObject {
  using table_t = zs::HashTable<int, 3, int, zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }
  table_t table;
};

struct ZenoIndexBuckets : IObject {
  using buckets_t = zs::IndexBuckets<3, int, int>;
  auto &get() noexcept { return ibs; }
  const auto &get() const noexcept { return ibs; }
  buckets_t ibs;
};

struct ZenoSparseLevelSet : IObject {
  using spls_t = zs::SparseLevelSet<3>;
  auto &get() noexcept { return ls; }
  const auto &get() const noexcept { return ls; }
  spls_t ls;
};

struct ZenoBoundary : IObject {
  using boundary_t =
      zs::variant<zs::LevelSetBoundary<zs::SparseLevelSet<3>>,
                  zs::Collider<zs::AnalyticLevelSet<
                      zs::analytic_geometry_e::Plane, float, 3>>>;
  enum category_e { LevelSet, Plane };

  auto &getBoundary() noexcept { return boundary; }
  const auto &getBoundary() const noexcept { return boundary; }
  template <category_e I> auto &getBoundary() noexcept {
    return std::get<I>(boundary);
  }
  template <category_e I> const auto &getBoundary() const noexcept {
    return std::get<I>(boundary);
  }
  boundary_t boundary;
};

} // namespace zeno