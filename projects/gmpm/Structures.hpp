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

struct ZpcInitializer {
  ZpcInitializer() {
    printf("Initializing Zpc resource\n");
    (void)zs::Resource::instance();
    printf("Initialized Zpc resource!\n");
  }
};

using ElasticModel =
    zs::variant<zs::FixedCorotated<float>, zs::NeoHookean<float>,
                zs::StvkWithHencky<float>>;
using AnisoElasticModel =
    zs::variant<std::monostate, zs::AnisotropicArap<float>>;
using PlasticModel =
    zs::variant<std::monostate, zs::NonAssociativeDruckerPrager<float>,
                zs::NonAssociativeVonMises<float>,
                zs::NonAssociativeCamClay<float>>;

struct ZenoConstitutiveModel : IObjectClone<ZenoConstitutiveModel> {
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

struct ZenoParticles : IObjectClone<ZenoParticles> {
  // (i  ) traditional mpm particle,
  // (ii ) lagrangian mesh vertex particle
  // (iii) lagrangian mesh element quadrature particle
  // tracker particle for
  enum category_e : int { mpm, curve, surface, tet, tracker, bending };
  using particles_t = zs::TileVector<float, 32>;

  ZenoParticles() = default;
  ~ZenoParticles() = default;
  ZenoParticles(ZenoParticles &&o) noexcept = default;
  ZenoParticles(const ZenoParticles &o)
      : elements{o.elements}, category{o.category}, prim{o.prim},
        model{o.model}, sprayedOffset{o.sprayedOffset}, asBoundary{
                                                            o.asBoundary} {
    if (o.particles)
      particles = std::make_shared<particles_t>(*o.particles);
  }
  ZenoParticles &operator=(ZenoParticles &&o) noexcept = default;
  ZenoParticles &operator=(const ZenoParticles &o) {
    ZenoParticles tmp{o};
    std::swap(*this, tmp);
    return *this;
  }

  auto &getParticles() {
    if (!particles)
      throw std::runtime_error("particles (verts) not inited.");
    return *particles;
  }
  const auto &getParticles() const {
    if (!particles)
      throw std::runtime_error("particles (verts) not inited.");
    return *particles;
  }
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
  bool isMeshPrimitive() const noexcept {
    return particles && elements.has_value() &&
           (category == category_e::curve || category == category_e::surface ||
            category == category_e::tet || category == category_e::tracker);
  }
  bool isLagrangianParticles() const noexcept {
    return particles && elements.has_value() &&
           (category == category_e::curve || category == category_e::surface ||
            category == category_e::tet);
  }
  auto &getModel() noexcept { return model; }
  const auto &getModel() const noexcept { return model; }

  int numDegree() const noexcept {
    switch (category) {
    case category_e::mpm:
      return 1;
    case category_e::curve:
      return 2;
    case category_e::surface:
      return 3;
    case category_e::tet:
      return 4;
    case category_e::tracker:
      return 0;
    default:
      return -1;
    }
    return -1;
  }
  std::size_t numParticles() const noexcept { return (*particles).size(); }
  std::size_t numElements() const noexcept { return (*elements).size(); }
  bool hasSprayedParticles() const noexcept {
    return sprayedOffset != numParticles();
  }

  std::shared_ptr<particles_t> particles{};
  std::optional<particles_t> elements{};
  category_e category{category_e::mpm}; // 0: conventional mpm particle, 1:
                                        // curve, 2: surface, 3: tet
  std::shared_ptr<PrimitiveObject> prim{};
  ZenoConstitutiveModel model{};
  std::size_t sprayedOffset{};
  bool asBoundary = false;
};

struct ZenoPartition : IObjectClone<ZenoPartition> {
  using Ti = int; // entry count
  using table_t = zs::HashTable<int, 3, Ti, zs::ZSPmrAllocator<false>>;
  using tag_t = zs::Vector<int>;
  using indices_t = zs::Vector<Ti>;

  auto &get() noexcept { return table; }
  const auto &get() const noexcept { return table; }

  auto numEntries() const noexcept { return table.size(); }
  auto numBoundaryEntries() const noexcept { return (*boundaryIndices).size(); }

  bool hasTags() const noexcept {
    return tags.has_value() && boundaryIndices.has_value();
  }
  auto &getTags() { return *tags; }
  const auto &getTags() const { return *tags; }
  auto &getBoundaryIndices() { return *boundaryIndices; }
  const auto &getBoundaryIndices() const { return *boundaryIndices; }

  void reserveTags() {
    auto numEntries = (std::size_t)table.size();
    if (!hasTags()) {
      tags = tag_t{numEntries, zs::memsrc_e::device, 0};
      boundaryIndices = indices_t{numEntries, zs::memsrc_e::device, 0};
    }
  }
  void clearTags() {
    if (hasTags())
      (*tags).reset(0);
  }

  table_t table;
  zs::optional<tag_t> tags;
  zs::optional<indices_t> boundaryIndices;
  bool requestRebuild{false};
  bool rebuilt;
};

struct ZenoGrid : IObjectClone<ZenoGrid> {
  enum transfer_scheme_e { Empty, Apic, Flip, AsFlip };
  using grid_t =
      zs::Grid<float, 3, 4, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return grid; }
  const auto &get() const noexcept { return grid; }

  bool isPicStyle() const noexcept { return transferScheme == "apic"; }
  bool hasPositionalAdjustment() const noexcept {
    return transferScheme == "asflip";
  }
  bool isFlipStyle() const noexcept {
    return transferScheme == "flip" || transferScheme == "aflip" ||
           transferScheme == "sflip" || transferScheme == "asflip";
  }
  bool isAffineAugmented() const noexcept {
    return transferScheme == "apic" || transferScheme == "aflip" ||
           transferScheme == "asflip";
  }

  grid_t grid;
  std::string transferScheme; //
  std::shared_ptr<ZenoPartition> partition;
};

struct ZenoIndexBuckets : IObjectClone<ZenoIndexBuckets> {
  using buckets_t = zs::IndexBuckets<3, int, int, zs::grid_e::collocated,
                                     zs::ZSPmrAllocator<false>>;
  auto &get() noexcept { return ibs; }
  const auto &get() const noexcept { return ibs; }
  buckets_t ibs;
};

struct ZenoLevelSet : IObjectClone<ZenoLevelSet> {
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
  using dummy_ls_t = typename basic_ls_t::dummy_ls_t;
  using uniform_vel_ls_t = typename basic_ls_t::uniform_vel_ls_t;

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
  decltype(auto) getSdfVelField() const noexcept {
    return std::get<const_sdf_vel_ls_t>(levelset);
  }
  decltype(auto) getSdfVelField() noexcept {
    return std::get<const_sdf_vel_ls_t>(levelset);
  }
  decltype(auto) getLevelSetSequence() const noexcept {
    return std::get<const_transition_ls_t>(levelset);
  }
  decltype(auto) getLevelSetSequence() noexcept {
    return std::get<const_transition_ls_t>(levelset);
  }
  template <zs::grid_e category = zs::grid_e::collocated>
  decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) const noexcept {
    return std::get<basic_ls_t>(levelset)
        .template getLevelSet<spls_t<category>>();
  }
  template <zs::grid_e category = zs::grid_e::collocated>
  decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) noexcept {
    return std::get<basic_ls_t>(levelset)
        .template getLevelSet<spls_t<category>>();
  }

  levelset_t levelset;
  std::string transferScheme;
};

struct ZenoBoundary : IObjectClone<ZenoBoundary> {
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