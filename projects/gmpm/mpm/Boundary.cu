#include "../Structures.hpp"
#include "../Utils.hpp"

#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/io/ParticleIO.hpp"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/simulation/Utils.hpp"
#include "zensim/zpc_tpls/fmt/color.h"
#include "zensim/zpc_tpls/fmt/format.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>

namespace zeno {

struct ApplyGridBoundaryOnZSGrid : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyGridBoundaryOnZSGrid\n");

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto bg = get_input<ZenoGrid>("BoundaryZSGrid");
    if (bg->transferScheme != "boundary")
      throw std::runtime_error("boundary grid is not of boundary type!");
    auto &boundaryGrid = get_input<ZenoGrid>("BoundaryZSGrid")->get();
    auto &boundaryPartition =
        get_input<ZenoPartition>("BoundaryZSPartition")->get();

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);
    auto typeStr = get_param<std::string>("type");
    collider_e type =
        typeStr == "sticky"
            ? collider_e::Sticky
            : (typeStr == "slip" ? collider_e::Slip : collider_e::Separate);

    const SmallString vtag = zsgrid->isFlipStyle() ? "vstar" : "v";
    cudaPol(Collapse{boundaryPartition.size(), boundaryGrid.block_size},
            [grid = proxy<execspace_e::cuda>({}, grid),
             boundaryGrid = proxy<execspace_e::cuda>({}, boundaryGrid),
             table = proxy<execspace_e::cuda>(partition),
             boundaryTable = proxy<execspace_e::cuda>(boundaryPartition), type,
             vtag] __device__(int bi, int ci) mutable {
              using table_t = RM_CVREF_T(table);
              auto boundaryBlock = boundaryGrid.block(bi);
              if (boundaryBlock("m", ci) == 0.f)
                return;
              auto blockid = boundaryTable._activeKeys[bi];
              auto blockno = table.query(blockid);
              if (blockno == table_t::sentinel_v)
                return;

              auto block = grid.block(blockno);
              if (block("m", ci) == 0.f)
                return;
              if (type == collider_e::Sticky)
                block.set(vtag, ci, boundaryBlock.pack<3>("v", ci));
              else {
                auto v_object = boundaryBlock.pack<3>("v", ci);
                auto normal = boundaryBlock.pack<3>("nrm", ci);
                auto v = block.pack<3>(vtag, ci);
                v -= v_object;
                auto proj = normal.dot(v);
                if ((type == collider_e::Separate && proj < 0) ||
                    type == collider_e::Slip)
                  v -= proj * normal;
                v += v_object;
                block.set(vtag, ci, v);
              }
            });

    fmt::print(fg(fmt::color::cyan),
               "done executing ApplyGridBoundaryOnZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyGridBoundaryOnZSGrid,
           {
               {"ZSPartition", "ZSGrid", "BoundaryZSPartition",
                "BoundaryZSGrid"},
               {"ZSGrid"},
               {{"enum sticky slip separate", "type", "sticky"}},
               {"MPM"},
           });

struct ApplyBoundaryOnZSGrid : INode {
  template <typename LsView>
  constexpr void
  projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                  const ZenoBoundary &boundary,
                  const typename ZenoPartition::table_t &partition,
                  typename ZenoGrid::grid_t &grid, bool isFlipStyle) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    if (isFlipStyle)
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               table = proxy<execspace_e::cuda>(partition),
               boundary = collider] __device__(int bi, int ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;

                  auto vstar = block.pack<3>("vstar", ci);
                  boundary.resolveCollision(pos, vstar);
                  block.set("vstar", ci, vstar);
                }
              });
    else
      cudaPol(Collapse{partition.size(), ZenoGrid::grid_t::block_space()},
              [grid = proxy<execspace_e::cuda>({}, grid),
               table = proxy<execspace_e::cuda>(partition),
               boundary = collider] __device__(int bi, int ci) mutable {
                auto block = grid.block(bi);
                auto mass = block("m", ci);
                if (mass != 0.f) {
                  auto vel = block.pack<3>("v", ci);
                  auto pos =
                      (table._activeKeys[bi] + grid.cellid_to_coord(ci)) *
                      grid.dx;
                  boundary.resolveCollision(pos, vel);
                  block.set("v", ci, vel);
                }
              });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyBoundaryOnZSGrid\n");

    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    if (has_input<ZenoBoundary>("ZSBoundary")) {
      auto boundary = get_input<ZenoBoundary>("ZSBoundary");
      auto &partition = get_input<ZenoPartition>("ZSPartition")->get();

      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      if (boundary->zsls)
        match([&](const auto &ls) {
          if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
            match([&](const auto &lsPtr) {
              auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
              projectBoundary(cudaPol, lsv, *boundary, partition, grid,
                              zsgrid->isFlipStyle());
            })(ls._ls);
          } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
            match([&](auto lsv) {
              projectBoundary(cudaPol, SdfVelFieldView{lsv}, *boundary,
                              partition, grid, zsgrid->isFlipStyle());
            })(ls.template getView<execspace_e::cuda>());
          } else if constexpr (is_same_v<RM_CVREF_T(ls),
                                         const_transition_ls_t>) {
            match([&](auto fieldPair) {
              auto &fvSrc = std::get<0>(fieldPair);
              auto &fvDst = std::get<1>(fieldPair);
              projectBoundary(cudaPol,
                              TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                                     SdfVelFieldView{fvDst},
                                                     ls._stepDt, ls._alpha},
                              *boundary, partition, grid,
                              zsgrid->isFlipStyle());
            })(ls.template getView<zs::execspace_e::cuda>());
          }
        })(boundary->zsls->getLevelSet());
    }

    fmt::print(fg(fmt::color::cyan), "done executing ApplyBoundaryOnZSGrid \n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyBoundaryOnZSGrid, {
                                      {"ZSPartition", "ZSGrid", "ZSBoundary"},
                                      {"ZSGrid"},
                                      {},
                                      {"MPM"},
                                  });

struct ComputeParticleBeta : INode {
  void determine_beta(zs::CudaExecutionPolicy &cudaPol,
                      typename ZenoParticles::particles_t &pars,
                      const typename ZenoPartition::table_t &partition,
                      const float dt, const typename ZenoGrid::grid_t &grid,
                      float JpcDefault = 1.f) {
    using namespace zs;
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 JpcDefault] __device__(size_t pi) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto pos = pars.pack<3>("x", pi);
      auto vp = pars.pack<3>("v", pi);
      pos += dt * vp;

      float Jp_critical = JpcDefault;

      auto arena =
          make_local_arena<grid_e::collocated, kernel_e::linear>(grid.dx, pos);

      bool movingIn = false;
      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0) // in case it is a partition for boundary grid
          continue;
        auto block = grid.block(blockno);

        const auto cellid = grid_t::coord_to_cellid(localIndex);
        // this cell already touched
        bool insideBoundary = block("m", cellid) != 0.f;
        if (insideBoundary) {
          auto boundaryVel = block.pack<3>("v", cellid);
          auto boundaryNrm = block.pack<3>("nrm", cellid);
          if ((vp - boundaryVel).dot(boundaryNrm) < 0) {
            movingIn = true;
            break;
          }
        }
      }
      if (movingIn)
        pars("beta", pi) = 0.f;
      else { // check critical volume ratio
        auto F = pars.pack<3, 3>("F", pi);
        auto J = determinant(F.template cast<double>());
        if (J > Jp_critical) // apply positional adjustment
          pars("beta", pi) = 0.5f;
        else
          pars("beta", pi) = 0.f; // assume beta_min = 0
      }
    });
  }
  void determine_beta(zs::CudaExecutionPolicy &cudaPol,
                      typename ZenoParticles::particles_t &pars,
                      typename ZenoParticles::particles_t &eles,
                      const typename ZenoPartition::table_t &partition,
                      const float dt, const typename ZenoGrid::grid_t &grid,
                      float JpcDefault = 1.f) {
    using namespace zs;
    cudaPol(range(eles.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 eles = proxy<execspace_e::cuda>({}, eles),
                                 table = proxy<execspace_e::cuda>(partition),
                                 grid = proxy<execspace_e::cuda>({}, grid), dt,
                                 JpcDefault] __device__(size_t ei) mutable {
      using grid_t = RM_CVREF_T(grid);
      auto pos = eles.pack<3>("x", ei);
      auto vp = eles.pack<3>("v", ei);
      pos += dt * vp;
      auto tri = eles.pack<3>("inds", ei).reinterpret_bits<int>();

      float Jp_critical = JpcDefault;

      auto arena =
          make_local_arena<grid_e::collocated, kernel_e::linear>(grid.dx, pos);

      bool movingIn = false;
      for (auto loc : arena.range()) {
        auto coord = arena.coord(loc);
        auto localIndex = coord & (grid_t::side_length - 1);
        auto blockno = table.query(coord - localIndex);
        if (blockno < 0) // in case it is a partition for boundary grid
          continue;
        auto block = grid.block(blockno);

        const auto cellid = grid_t::coord_to_cellid(localIndex);
        // this cell already touched
        bool insideBoundary = block("m", cellid) != 0.f;
        if (insideBoundary) {
          auto boundaryVel = block.pack<3>("v", cellid);
          auto boundaryNrm = block.pack<3>("nrm", cellid);
          if ((vp - boundaryVel).dot(boundaryNrm) < 0) {
            movingIn = true;
            break;
          }
        }
      }
      if (movingIn)
        for (int vi = 0; vi != 3; ++vi)
          pars("beta", tri[vi]) = 0.f;
      else { // check critical volume ratio
#if 0
        auto d = eles.pack<3, 3>("d", ei);
        auto [Q, R] = math::gram_schmidt(d);
        if (R(2, 2) > 0.999f) // apply positional adjustment
#else
        auto F = eles.pack<3, 3>("F", ei);
        auto J = determinant(F);  // area ratio * nrm ratio
        auto d = eles.pack<3, 3>("d", ei);
        auto J0inv = zs::abs(determinant(eles.pack<3, 3>("DmInv", ei)));
        auto [Q, R] = math::gram_schmidt(d);
        auto Jn = zs::abs(R(0, 0) * R(1, 1) - R(1, 0) * R(0, 1));
        J /= (Jn * J0inv);  // normal direction
        if (J > Jp_critical) // apply positional adjustment
#endif
        for (int vi = 0; vi != 3; ++vi)
          pars("beta", tri[vi]) = 0.2f;
        else for (int vi = 0; vi != 3; ++vi) pars("beta", tri[vi]) = 0.f;
      }
    });
  }
  template <typename LsView>
  void determine_beta_sdf(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                          const ZenoBoundary &boundary, const float dt,
                          typename ZenoParticles::particles_t &pars,
                          float JpcDefault = 1.f) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    cudaPol(range(pars.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 boundary = collider, JpcDefault,
                                 dt] __device__(size_t pi) mutable {
      auto pos = pars.pack<3>("x", pi);
      auto vp = pars.pack<3>("v", pi);
      pos += dt * vp;

      float Jp_critical = JpcDefault;

      auto X = boundary.R.transpose() * (pos - boundary.b) / boundary.s;
      if (boundary.levelset.getSignedDistance(X) < 0) {
        auto nrm = boundary.R * boundary.levelset.getNormal(X);
        auto vel = boundary.R * boundary.levelset.getMaterialVelocity(X) +
                   boundary.dbdt;
        if ((vp - vel).dot(nrm) < 0) {
          pars("beta", pi) = 0.f;
          return;
        }
      }
      // check critical volume ratio
      auto F = pars.pack<3, 3>("F", pi);
      auto J = determinant(F.template cast<double>());
      if (J > Jp_critical) // apply positional adjustment
        pars("beta", pi) = 0.5f;
      else
        pars("beta", pi) = 0.f; // assume beta_min = 0
    });
  }
  template <typename LsView>
  void determine_beta_sdf(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                          const ZenoBoundary &boundary, const float dt,
                          typename ZenoParticles::particles_t &pars,
                          typename ZenoParticles::particles_t &eles,
                          float JpcDefault = 1.f) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    cudaPol(range(eles.size()), [pars = proxy<execspace_e::cuda>({}, pars),
                                 eles = proxy<execspace_e::cuda>({}, eles),
                                 boundary = collider, JpcDefault,
                                 dt] __device__(size_t ei) mutable {
      auto pos = eles.pack<3>("x", ei);
      auto vp = eles.pack<3>("v", ei);
      pos += dt * vp;
      auto tri = eles.pack<3>("inds", ei).reinterpret_bits<int>();

      float Jp_critical = JpcDefault;

      auto X = boundary.R.transpose() * (pos - boundary.b) / boundary.s;
      if (boundary.levelset.getSignedDistance(X) < 0) {
        auto nrm = boundary.R * boundary.levelset.getNormal(X);
        auto vel = boundary.R * boundary.levelset.getMaterialVelocity(X) +
                   boundary.dbdt;
        if ((vp - vel).dot(nrm) < 0) {
          for (int vi = 0; vi != 3; ++vi)
            pars("beta", tri[vi]) = 0.f;
          return;
        }
      }
// check critical volume ratio
#if 0
      auto d = eles.pack<3, 3>("d", ei);
      auto [Q, R] = math::gram_schmidt(d);
      if (R(2, 2) > 0.999f) // apply positional adjustment
#else
      auto F = eles.pack<3, 3>("F", ei);
      auto J = determinant(F);  // area ratio * nrm ratio
      auto d = eles.pack<3, 3>("d", ei);
      auto J0inv = zs::abs(determinant(eles.pack<3, 3>("DmInv", ei)));
      auto [Q, R] = math::gram_schmidt(d);
      auto Jn = zs::abs(R(0, 0) * R(1, 1) - R(1, 0) * R(0, 1));
      J /= (Jn * J0inv);  // normal direction
      if (J > Jp_critical) // apply positional adjustment
#endif
      for (int vi = 0; vi != 3; ++vi)
        pars("beta", tri[vi]) = 0.2f;
      else for (int vi = 0; vi != 3; ++vi) pars("beta", tri[vi]) = 0.f;
    });
  }

  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing ComputeParticleBeta\n");

    using namespace zs;
    auto cudaPol = cuda_exec().device(0);
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto stepDt = get_input<zeno::NumericObject>("dt")->get<float>();

    if (has_input<ZenoGrid>("ZSBoundary(Grid)")) {
      auto zsgrid = get_input<ZenoGrid>("ZSBoundary(Grid)");
      auto &grid = zsgrid->get();

      if (zsgrid->transferScheme != "boundary")
        throw std::runtime_error("the grid passed in for computing particle "
                                 "beta should represent a boundary!");

      for (auto &&parObjPtr : parObjPtrs) {
        auto &pars = parObjPtr->getParticles();
        auto &model = parObjPtr->getModel();

        fmt::print("[ComputeParticleBeta] dx: {}, dt: {}, npars: {}\n", grid.dx,
                   stepDt, pars.size());

        if (parObjPtr->category == ZenoParticles::mpm ||
            parObjPtr->category == ZenoParticles::surface) {
          bool isMeshPrimitive = parObjPtr->isMeshPrimitive();
          match([&](auto &elasticModel, auto &anisoElasticModel) {
            float JpcDefault = 1.f;
            if constexpr (is_same_v<RM_CVREF_T(anisoElasticModel),
                                    NonAssociativeCamClay<float>>)
              JpcDefault = 1.1f;
            if (isMeshPrimitive)
              determine_beta(cudaPol, pars, parObjPtr->getQuadraturePoints(),
                             partition, stepDt, grid, JpcDefault);
            else
              determine_beta(cudaPol, pars, partition, stepDt, grid,
                             JpcDefault);
          })(model.getElasticModel(), model.getAnisoElasticModel());
        }
      }
    } else if (has_input<ZenoBoundary>("ZSBoundary(Grid)")) {
      auto boundary = get_input<ZenoBoundary>("ZSBoundary(Grid)");
      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      for (auto &&parObjPtr : parObjPtrs) {
        auto &pars = parObjPtr->getParticles();
        auto &model = parObjPtr->getModel();

        if (parObjPtr->category == ZenoParticles::mpm ||
            parObjPtr->category == ZenoParticles::surface) {
          bool isMeshPrimitive = parObjPtr->isMeshPrimitive();
          float JpcDefault = 1.f;
          match([&](auto &elasticModel, auto &anisoElasticModel) {
            if constexpr (is_same_v<RM_CVREF_T(anisoElasticModel),
                                    NonAssociativeCamClay<float>>)
              JpcDefault = 1.1f;
          })(model.getElasticModel(), model.getAnisoElasticModel());

          if (boundary->zsls)
            match([&](const auto &ls) {
              if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
                match([&](const auto &lsPtr) {
                  auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
                  if (isMeshPrimitive)
                    determine_beta_sdf(cudaPol, lsv, *boundary, stepDt, pars,
                                       parObjPtr->getQuadraturePoints(),
                                       JpcDefault);
                  else
                    determine_beta_sdf(cudaPol, lsv, *boundary, stepDt, pars,
                                       JpcDefault);
                })(ls._ls);
              } else if constexpr (is_same_v<RM_CVREF_T(ls),
                                             const_sdf_vel_ls_t>) {
                match([&](auto lsv) {
                  if (isMeshPrimitive)
                    determine_beta_sdf(
                        cudaPol, SdfVelFieldView{lsv}, *boundary, stepDt, pars,
                        parObjPtr->getQuadraturePoints(), JpcDefault);
                  else
                    determine_beta_sdf(cudaPol, SdfVelFieldView{lsv}, *boundary,
                                       stepDt, pars, JpcDefault);
                })(ls.template getView<execspace_e::cuda>());
              } else if constexpr (is_same_v<RM_CVREF_T(ls),
                                             const_transition_ls_t>) {
                match([&](auto fieldPair) {
                  auto &fvSrc = std::get<0>(fieldPair);
                  auto &fvDst = std::get<1>(fieldPair);
                  if (isMeshPrimitive)
                    determine_beta_sdf(
                        cudaPol,
                        TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                               SdfVelFieldView{fvDst},
                                               ls._stepDt, ls._alpha},
                        *boundary, stepDt, pars,
                        parObjPtr->getQuadraturePoints(), JpcDefault);
                  else
                    determine_beta_sdf(
                        cudaPol,
                        TransitionLevelSetView{SdfVelFieldView{fvSrc},
                                               SdfVelFieldView{fvDst},
                                               ls._stepDt, ls._alpha},
                        *boundary, stepDt, pars, JpcDefault);
                })(ls.template getView<zs::execspace_e::cuda>());
              }
            })(boundary->zsls->getLevelSet());
        }
      }
    } else {
      for (auto &&parObjPtr : parObjPtrs) {
        auto &pars = parObjPtr->getParticles();
        auto &model = parObjPtr->getModel();

        if (parObjPtr->category == ZenoParticles::mpm ||
            parObjPtr->category == ZenoParticles::surface) {
          bool isMeshPrimitive = parObjPtr->isMeshPrimitive();
          float JpcDefault = 1.f;
          match([&](auto &elasticModel, auto &anisoElasticModel) {
            if constexpr (is_same_v<RM_CVREF_T(anisoElasticModel),
                                    NonAssociativeCamClay<float>>)
              JpcDefault = 1.1f;
          })(model.getElasticModel(), model.getAnisoElasticModel());
          if (isMeshPrimitive) {
            auto &eles = parObjPtr->getQuadraturePoints();
            cudaPol(range(eles.size()),
                    [pars = proxy<execspace_e::cuda>({}, pars),
                     eles = proxy<execspace_e::cuda>({}, eles),
                     Jp_critical = JpcDefault] __device__(size_t ei) mutable {
                      auto tri =
                          eles.pack<3>("inds", ei).reinterpret_bits<int>();

                      auto F = eles.pack<3, 3>("F", ei);
                      auto J = determinant(F); // area ratio * nrm ratio
                      auto d = eles.pack<3, 3>("d", ei);
                      auto J0inv =
                          zs::abs(determinant(eles.pack<3, 3>("DmInv", ei)));
                      auto [Q, R] = math::gram_schmidt(d);
                      auto Jn = zs::abs(R(0, 0) * R(1, 1) - R(1, 0) * R(0, 1));
                      J /= (Jn * J0inv);   // normal direction
                      if (J > Jp_critical) // apply positional adjustment
                        for (int vi = 0; vi != 3; ++vi)
                          pars("beta", tri[vi]) = 0.2f;
                      else
                        for (int vi = 0; vi != 3; ++vi)
                          pars("beta", tri[vi]) = 0.f;
                    });
          } else {
            cudaPol(range(pars.size()),
                    [pars = proxy<execspace_e::cuda>({}, pars),
                     Jp_critical = JpcDefault] __device__(size_t pi) mutable {
                      auto F = pars.pack<3, 3>("F", pi);
                      auto J = determinant(F); // area ratio * nrm ratio
                      if (J > Jp_critical)     // apply positional adjustment
                        pars("beta", pi) = 0.5f;
                      else
                        pars("beta", pi) = 0.f;
                    });
          }
        }
      }
    }

    fmt::print(fg(fmt::color::cyan), "done executing ComputeParticleBeta\n");
    set_output("ZSParticles", get_input("ZSParticles"));
  }
};

ZENDEFNODE(ComputeParticleBeta,
           {
               {"ZSParticles", "ZSPartition", "ZSBoundary(Grid)", "dt"},
               {"ZSParticles"},
               {},
               {"MPM"},
           });

struct ApplyWindImpulseOnZSGrid : INode {
  template <typename VelSplsViewT>
  void computeWindImpulse(zs::CudaExecutionPolicy &cudaPol, float windDragCoeff,
                          float windDensity, VelSplsViewT velLs,
                          const typename ZenoParticles::particles_t &pars,
                          const typename ZenoParticles::particles_t &eles,
                          const typename ZenoPartition::table_t &partition,
                          typename ZenoGrid::grid_t &grid, float dt) {
    using namespace zs;
    cudaPol(
        range(eles.size()),
        [windDragCoeff, windDensity, velLs,
         pars = proxy<execspace_e::cuda>({}, pars), // for normal compute
         eles = proxy<execspace_e::cuda>({}, eles),
         table = proxy<execspace_e::cuda>(partition),
         grid = proxy<execspace_e::cuda>({}, grid),
         Dinv = 4.f / grid.dx / grid.dx, dt] __device__(size_t ei) mutable {
          using grid_t = RM_CVREF_T(grid);
          zs::vec<float, 3> n{};
          float area{};
          {
            auto p0 =
                pars.pack<3>("x", reinterpret_bits<int>(eles("inds", 0, ei)));
            auto p1 =
                pars.pack<3>("x", reinterpret_bits<int>(eles("inds", 1, ei)));
            auto p2 =
                pars.pack<3>("x", reinterpret_bits<int>(eles("inds", 2, ei)));
            auto cp = (p1 - p0).cross(p2 - p0);
            area = cp.length();
            n = cp / area;
            area *= 0.5f;
          }
          auto pos = eles.pack<3>("x", ei);
          auto windVel = velLs.getMaterialVelocity(pos);

          auto vel = eles.pack<3>("v", ei);
          auto vrel = windVel - vel;
          float vnSignedLength = n.dot(vrel);
          auto vn = n * vnSignedLength;
          auto vt = vrel - vn; // tangent
          auto windForce = windDensity * area * zs::abs(vnSignedLength) * vn +
                           windDragCoeff * area * vt;
          auto fdt = windForce * dt;

          auto arena =
              make_local_arena<grid_e::collocated, kernel_e::quadratic>(grid.dx,
                                                                        pos);

          for (auto loc : arena.range()) {
            auto coord = arena.coord(loc);
            auto localIndex = coord & (grid_t::side_length - 1);
            auto blockno = table.query(coord - localIndex);
            if (blockno < 0)
              printf("THE HELL!");
            auto block = grid.block(blockno);
            auto W = arena.weight(loc);
            const auto cellid = grid_t::coord_to_cellid(localIndex);
            for (int d = 0; d != 3; ++d)
              atomic_add(exec_cuda, &block("v", d, cellid), W * fdt[d]);
          }
        });
  }
  void apply() override {
    fmt::print(fg(fmt::color::green),
               "begin executing ApplyWindImpulseOnZSGrid\n");

    using namespace zs;

    // this could possibly be the same staggered velocity field too
    auto parObjPtrs = RETRIEVE_OBJECT_PTRS(ZenoParticles, "ZSParticles");
    auto &partition = get_input<ZenoPartition>("ZSPartition")->get();
    auto zsgrid = get_input<ZenoGrid>("ZSGrid");
    auto &grid = zsgrid->get();

    auto velZsField = get_input<ZenoLevelSet>("ZSVelField");
    const auto &velField = velZsField->getBasicLevelSet()._ls;

    auto stepDt = get_input2<float>("dt");
    auto windDrag = get_input2<float>("windDrag");
    auto windDensity = get_input2<float>("windDensity");

    match([&](const auto &velLsPtr) {
      auto cudaPol = cuda_exec().device(0);
      for (auto &&parObjPtr : parObjPtrs) {
        auto &pars = parObjPtr->getParticles();
        if (parObjPtr->category == ZenoParticles::surface ||
            parObjPtr->category == ZenoParticles::tracker) {
          auto &eles = parObjPtr->getQuadraturePoints();
          computeWindImpulse(cudaPol, windDrag, windDensity,
                             get_level_set_view<execspace_e::cuda>(velLsPtr),
                             pars, eles, partition, grid, stepDt);
        }
      }
    })(velField);

    fmt::print(fg(fmt::color::cyan),
               "done executing ApplyWindImpulseOnZSGrid\n");
    set_output("ZSGrid", zsgrid);
  }
};

ZENDEFNODE(ApplyWindImpulseOnZSGrid, {
                                         {"ZSParticles",
                                          "ZSVelField",
                                          "ZSPartition",
                                          "ZSGrid",
                                          {"float", "windDrag", "0"},
                                          {"float", "windDensity", "1"},
                                          {"float", "dt", "0.1"}},
                                         {"ZSGrid"},
                                         {},
                                         {"MPM"},
                                     });

struct TransformZSLevelSet : INode {
  void apply() override {
    fmt::print(fg(fmt::color::green), "begin executing TransformZSLevelSet\n");
    auto zsls = get_input<ZenoLevelSet>("ZSLevelSet");
    auto &ls = zsls->getLevelSet();

    using namespace zs;
    using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
    // translation
    if (has_input("translation")) {
      auto b = get_input<NumericObject>("translation")->get<vec3f>();
      match(
          [&b](basic_ls_t &basicLs) {
            match(
                [b](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [b](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [b](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->translate(zs::vec<float, 3>{b[0], b[1], b[2]});
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }

    // scale
    if (has_input("scaling")) {
      auto s = get_input<NumericObject>("scaling")->get<float>();
      match(
          [&s](basic_ls_t &basicLs) {
            match(
                [s](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [s](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [s](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->scale(s);
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }
    // rotation
    if (has_input("eulerXYZ")) {
      auto yprAngles = get_input<NumericObject>("eulerXYZ")->get<vec3f>();
      auto rot = zs::Rotation<float, 3>{yprAngles[0], yprAngles[1],
                                        yprAngles[2], zs::degree_c, zs::ypr_c};
      match(
          [&rot](basic_ls_t &basicLs) {
            match(
                [rot](std::shared_ptr<typename basic_ls_t::clspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [rot](std::shared_ptr<typename basic_ls_t::ccspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [rot](std::shared_ptr<typename basic_ls_t::sgspls_t> lsPtr) {
                  lsPtr->rotate(rot.transpose());
                },
                [](auto &lsPtr) {
                  auto msg = get_var_type_str(*lsPtr);
                  throw std::runtime_error(fmt::format(
                      "levelset of type [{}] cannot be transformed yet.", msg));
                })(basicLs._ls);
          },
          [](auto &ls) {
            auto msg = get_var_type_str(ls);
            throw std::runtime_error(
                fmt::format("levelset of special type [{}] are const-.", msg));
          })(ls);
    }

    fmt::print(fg(fmt::color::cyan), "done executing TransformZSLevelSet\n");
    set_output("ZSLevelSet", zsls);
  }
};
// refer to nodes/prim/TransformPrimitive.cpp
ZENDEFNODE(TransformZSLevelSet,
           {
               {"ZSLevelSet", "translation", "eulerXYZ", "scaling"},
               {"ZSLevelSet"},
               {},
               {"MPM"},
           });

} // namespace zeno