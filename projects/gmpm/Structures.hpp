#pragma once
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/physics/constitutive_models/AnisotropicArap.hpp"
#include "zensim/physics/constitutive_models/EquationOfState.hpp"
#include "zensim/physics/constitutive_models/FixedCorotated.h"
#include "zensim/physics/constitutive_models/NeoHookean.hpp"
#include "zensim/physics/constitutive_models/StvkWithHencky.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeCamClay.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeDruckerPrager.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeVonMises.hpp"
#include "zensim/resource/Resource.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>

namespace zeno {

using ElasticModel =
    zs::variant<zs::FixedCorotated<float>, zs::NeoHookean<float>,
                zs::StvkWithHencky<float>>;
using AnisoElasticModel =
    zs::variant<std::monostate, zs::AnisotropicArap<float>>;
using PlasticModel =
    zs::variant<std::monostate, zs::NonAssociativeDruckerPrager<float>,
                zs::NonAssociativeVonMises<float>,
                zs::NonAssociativeCamClay<float>>;

struct ZenoConstitutiveModel : IObject {
  enum elastic_model_e { Fcr, Nhk, Stvk };
  enum aniso_plastic_model_e { None_, Arap };
  enum plastic_model_e { None, DruckerPrager, VonMises, CamClay };

  auto &getElasticModel() noexcept { return elasticModel; }
  const auto &getElasticModel() const noexcept { return elasticModel; }
  auto &getAnisoElasticModel() noexcept { return anisoElasticModel; }
  const auto &getAnisoElasticModel() const noexcept {
    return anisoElasticModel;
  }
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

  bool hasAnisoModel() const noexcept {
    return !std::holds_alternative<std::monostate>(anisoElasticModel);
  }
  bool hasPlasticModel() const noexcept {
    return !std::holds_alternative<std::monostate>(plasticModel);
  }

  bool hasF() const noexcept { return elasticModel.index() < 3; }
  bool hasLogJp() const noexcept { return plasticModel.index() == CamClay; }
  bool hasOrientation() const noexcept {
    return anisoElasticModel.index() != None;
  }
  bool hasPlasticity() const noexcept { return plasticModel.index() != None; }

  float dx, volume, density;
  ElasticModel elasticModel;
  AnisoElasticModel anisoElasticModel;
  PlasticModel plasticModel;
};

struct ZenoParticles : IObject {
  // (i  ) traditional mpm particle,
  // (ii ) lagrangian mesh vertex particle
  // (iii) lagrangian mesh element quadrature particle
  enum category_e : int { mpm, curve, surface, tet };
  using particles_t =
      zs::TileVector<float, 32, unsigned char, zs::ZSPmrAllocator<false>>;
  auto &getParticles() noexcept { return particles; }
  const auto &getParticles() const noexcept { return particles; }
  auto &getQuadraturePoints() {
    if (!elements.has_value())
      throw std::runtime_error("quadrature points not binded.");
    return *elements;
  }
  const auto &getQuadraturePoints() const {
    if (!elements.has_value())
      throw std::runtime_error("quadrature points not binded.");
    return *elements;
  }
  auto &getModel() noexcept { return model; }
  const auto &getModel() const noexcept { return model; }

  particles_t particles{};
  std::optional<particles_t> elements{};
  category_e category{category_e::mpm}; // 0: conventional mpm particle, 1:
                                        // curve, 2: surface, 3: tet
  std::shared_ptr<PrimitiveObject> prim;
  ZenoConstitutiveModel model{};
};

struct ZenoPartition : IObject {
  using table_t = zs::HashTable<int, 3, int, zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }
  table_t table;
};

struct ZenoGrid : IObject {
  enum transfer_scheme_e { Empty, Apic, Flip, AsFlip };
  using grid_t =
      zs::Grid<float, 3, 4, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return grid; }
  const auto &get() const noexcept { return grid; }

  grid_t grid;
  std::string transferScheme; //
  std::shared_ptr<ZenoPartition> partition;
};

struct ZenoIndexBuckets : IObject {
  using buckets_t = zs::IndexBuckets<3, int, int, zs::grid_e::collocated,
                                     zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return ibs; }
  const auto &get() const noexcept { return ibs; }
  buckets_t ibs;
};

struct ZenoLevelSet : IObject {
  // this supports a wide range of grid types (not just collocated)
  // default channel contains "sdf"
  // default transfer scheme is "unknown"
  using basic_ls_t = zs::BasicLevelSet<float, 3>;
  using const_sdf_vel_ls_t = zs::ConstSdfVelFieldPtr<float, 3>;
  using const_transition_ls_t = zs::ConstTransitionLevelSetPtr<float, 3>;
  using levelset_t =
      zs::variant<basic_ls_t, const_sdf_vel_ls_t, const_transition_ls_t>;

  template <zs::grid_e category = zs::grid_e::collocated>
  using spls_t = typename basic_ls_t::template spls_t<category>;
  using clspls_t = typename basic_ls_t::clspls_t;
  using ccspls_t = typename basic_ls_t::ccspls_t;
  using sgspls_t = typename basic_ls_t::sgspls_t;

  auto &getLevelSet() noexcept { return levelset; }
  const auto &getLevelSet() const noexcept { return levelset; }

  bool holdsBasicLevelSet() const noexcept {
    return std::holds_alternative<basic_ls_t>(levelset);
  }
  template <zs::grid_e category = zs::grid_e::collocated>
  bool holdsSparseLevelSet(zs::wrapv<category> = {}) const noexcept {
    return zs::match([](const auto &ls) {
      if constexpr (zs::is_same_v<RM_CVREF_T(ls), basic_ls_t>)
        return ls.template holdsLevelSet<spls_t<category>>();
      else
        return false;
    })(levelset);
  }
  decltype(auto) getBasicLevelSet() const noexcept {
    return std::get<basic_ls_t>(levelset);
  }
  decltype(auto) getBasicLevelSet() noexcept {
    return std::get<basic_ls_t>(levelset);
  }
  decltype(auto) getLevelSetSequence() const noexcept {
    return std::get<const_transition_ls_t>(levelset);
  }
  decltype(auto) getLevelSetSequence() noexcept {
    return std::get<const_transition_ls_t>(levelset);
  }
  template <zs::grid_e category = zs::grid_e::collocated>
  decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) const noexcept {
    return std::get<basic_ls_t>(levelset).getLevelSet<spls_t<category>>();
  }
  template <zs::grid_e category = zs::grid_e::collocated>
  decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) noexcept {
    return std::get<basic_ls_t>(levelset).getLevelSet<spls_t<category>>();
  }

  levelset_t levelset;
  std::string transferScheme;
};

struct ZenoBoundary : IObject {
  using levelset_t = typename ZenoLevelSet::levelset_t;

  template <typename LsView> auto getBoundary(LsView &&lsv) const noexcept {
    using namespace zs;
    auto ret = Collider{FWD(lsv), type};
    ret.s = s;
    ret.dsdt = dsdt;
    ret.R = R;
    ret.omega = omega;
    ret.b = b;
    ret.dbdt = dbdt;
    return ret;
  }

  // levelset_t *levelset{nullptr};
  std::shared_ptr<ZenoLevelSet> zsls{};
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