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

  float volume, density;
  ElasticModel elasticModel;
  AnisoElasticModel anisoElasticModel;
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
  using buckets_t = zs::IndexBuckets<3, int, int, zs::grid_e::collocated,
                                     zs::ZSPmrAllocator<false>>;
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
#if 0
  using sdf_vel_t = zs::ConstSdfVelFieldPtr<float, 3>;
  using transition_ls_t = zs::ConstTransitionLevelSetPtr<float, 3>;
  void test() {
    zs::SparseLevelSetView<zs::execspace_e::cuda,
                           zs::SparseLevelSet<3, zs::grid_e::collocated>, void>
        a{};
    zs::BasicLevelSet<float, 3> x{zs::DummyLevelSet<float, 3>{}};
    zs::BasicLevelSet<float, 3> y{
        std::make_shared<zs::DummyLevelSet<float, 3>>()};
    // zs::name_that_type(typename sdf_vel_t::template sdf_vel_ls_view_t<
    //                   zs::execspace_e::cuda>{});
    sdf_vel_t z{y};
    transition_ls_t zz{};
    zz.push(x);
    zz.push(z);

#if 1
#if 1
    auto [lsvSrc, lsvDst] = zz.getView<zs::execspace_e::cuda>();
    auto srcStr = zs::get_var_type_str(lsvSrc);
    auto dstStr = zs::get_var_type_str(lsvDst);
    zs::match([srcStr, dstStr](auto a, auto b) {
      fmt::print("lsv src: [{}] ({}), \n\nlsv dst: [{}] ({})\n",
                 zs::get_var_type_str(a), srcStr, zs::get_var_type_str(b),
                 dstStr);
    })(lsvSrc, lsvDst);
#endif
#endif
  }
#endif

  using basic_ls_t = zs::BasicLevelSet<float, 3>;
  using spls_t = typename basic_ls_t::spls_t;
  using sdf_vel_ls_t = zs::ConstSdfVelFieldPtr<float, 3>;
  using transition_ls_t = zs::ConstTransitionLevelSetPtr<float, 3>;
  using levelset_t = zs::variant<basic_ls_t, sdf_vel_ls_t, transition_ls_t>;

  auto &getLevelSet() noexcept { return levelset; }
  const auto &getLevelSet() const noexcept { return levelset; }

  bool holdsBasicLevelSet() const noexcept {
    return std::holds_alternative<basic_ls_t>(levelset);
  }
  bool holdsSparseLevelSet() const noexcept {
    return zs::match([](const auto &ls) {
      if constexpr (zs::is_same_v<RM_CVREF_T(ls), basic_ls_t>)
        return ls.template holdsLevelSet<typename basic_ls_t::spls_t>();
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
    return std::get<transition_ls_t>(levelset);
  }
  decltype(auto) getLevelSetSequence() noexcept {
    return std::get<transition_ls_t>(levelset);
  }
  decltype(auto) getSparseLevelSet() const noexcept {
    return std::get<basic_ls_t>(levelset)
        .getLevelSet<typename basic_ls_t::spls_t>();
  }
  decltype(auto) getSparseLevelSet() noexcept {
    return std::get<basic_ls_t>(levelset)
        .getLevelSet<typename basic_ls_t::spls_t>();
  }

  levelset_t levelset;
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

  levelset_t *levelset{nullptr};
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