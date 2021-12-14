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

struct ZenoLevelSet : IObject {
  using spls_t = zs::SparseLevelSet<3>;
  using plane_t =
      zs::AnalyticLevelSet<zs::analytic_geometry_e::Plane, float, 3>;
  using cuboid_t =
      zs::AnalyticLevelSet<zs::analytic_geometry_e::Cuboid, float, 3>;
  using sphere_t =
      zs::AnalyticLevelSet<zs::analytic_geometry_e::Sphere, float, 3>;
  using levelset_t = zs::variant<spls_t, plane_t, cuboid_t, sphere_t>;
  enum category_e { LevelSet, Plane, Cuboid, Sphere };

  auto &getLevelSet() noexcept { return levelset; }
  const auto &getLevelSet() const noexcept { return levelset; }

  bool holdSparseLevelSet() const noexcept {
    return std::holds_alternative<spls_t>(levelset);
  }

  decltype(auto) getSparseLevelSet() noexcept {
    return std::get<spls_t>(levelset);
  }
  decltype(auto) getSparseLevelSet() const noexcept {
    return std::get<spls_t>(levelset);
  }

  template <category_e I> auto &getLevelSet() noexcept {
    return std::get<I>(levelset);
  }
  template <category_e I> const auto &getLevelSet() const noexcept {
    return std::get<I>(levelset);
  }

  levelset_t levelset;
};

struct ZenoBoundary : IObject {
  using spls_t = typename ZenoLevelSet::spls_t;
  using plane_t = typename ZenoLevelSet::plane_t;
  using cuboid_t = typename ZenoLevelSet::cuboid_t;
  using sphere_t = typename ZenoLevelSet::sphere_t;
  using levelset_t = typename ZenoLevelSet::levelset_t;
  using category_e = typename ZenoLevelSet::category_e;

  auto &getSdfField() noexcept { return *levelset; }
  const auto &getSdfField() const noexcept { return *levelset; }
  auto &getVelocityField() noexcept { return *velocityField; }
  const auto &getVelocityField() const noexcept { return *velocityField; }
  bool hasVelocityField() const noexcept { return velocityField != nullptr; }

  template <typename LS> auto getLevelSetView(LS &&ls) const noexcept {
    using LsT = zs::remove_cvref_t<LS>;
    if constexpr (zs::is_same_v<LsT, spls_t>) {
      return zs::proxy<zs::execspace_e::cuda>(FWD(ls));
    } else
      return FWD(ls);
  }

  template <typename LsView> auto getBoundary(LsView &&lsv) const noexcept {
    using namespace zs;
    auto ret = Collider{lsv, type};
    ret.s = s;
    ret.dsdt = dsdt;
    ret.R = R;
    ret.omega = omega;
    ret.b = b;
    ret.dbdt = dbdt;
    return ret;
  }

  levelset_t *levelset{nullptr};
  levelset_t *velocityField{nullptr};
  zs::collider_e type{zs::collider_e::Sticky};
  /** scale **/
  float s{1};
  float dsdt{0};
  /** rotation **/
  zs::Rotation<float, 3> R{zs::Rotation<float, 3>::identity()};
  zs::AngularVelocity<float, 3> omega{};
  /** translation **/
  zs::vec<float, 3> b{zs::vec<float, 3>::zeros()};
  zs::vec<float, 3> dbdt{zs::vec<float, 3>::zeros()};
};

} // namespace zeno