#include "../Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ApplyBoundaryOnVertices : INode {
  template <typename LsView, typename TileVecT>
  constexpr void projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                                 const ZenoBoundary &boundary, TileVecT &verts,
                                 float dt) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts), boundary = collider,
             dt] __device__(int vi) mutable {
              using mat3 = zs::vec<double, 3, 3>;
              auto vel = verts.template pack<3>("v", vi);
              auto pos = verts.template pack<3>("x", vi);
              if (boundary.queryInside(pos)) {
                auto v = boundary.getVelocity(pos);
                auto n = boundary.getNormal(pos);
                if (v.l2NormSqr() == 0)
                  verts("BCfixed", vi) = 1;
                if (boundary.type == collider_e::Sticky) {
                  verts.template tuple<9>("BCbasis", vi) = mat3::identity();
                  verts("BCorder", vi) = 3;
                  verts.template tuple<3>("BCtarget", vi) =
                      verts.template pack<3>("BCtarget", vi) + dt * v;
                } else if (boundary.type == collider_e::Slip) {
                  auto BCbasis = verts.template pack<3, 3>("BCbasis", vi);
                  auto BCtarget = verts.template pack<3>("BCtarget", vi);
                  int BCorder = verts("BCorder", vi);
                  if (BCorder >= 3)
                    return;
                  for (int d = 0; d != BCorder; ++d) {
                    auto nd = col(BCbasis, d);
                    // remove components in previous normal directions
                    n -= n.dot(nd) * nd;
                  }
                  if (n.l2NormSqr() > limits<float>::epsilon()) {
                    n = n.normalized();
                    for (int d = 0; d != 3; ++d)
                      BCbasis(d, BCorder) = n(d);
                    verts.template tuple<9>("BCbasis", vi) = BCbasis;
                    verts("BCorder", vi) = (BCorder + 1);
                    verts.template tuple<3>("BCtarget", vi) =
                        verts.template pack<3>("BCtarget", vi) +
                        dt * v.dot(n) * n;
                  }
                }
              }
            });
  }
  void apply() override {
    auto zsverts = get_input<ZenoParticles>("ZSParticles");
    auto &verts = zsverts->getParticles<true>();
    auto dt = get_input2<float>("dt");

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    /// init BC status
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts)] __device__(
                int vi) mutable {
              using mat3 = zs::vec<zs::f64, 3, 3>;
              verts.tuple<9>("BCbasis", vi) = mat3::identity();
              verts("BCorder", vi) = 0;
              verts("BCfixed", vi) = 0;
              verts.tuple<3>("BCtarget", vi) = verts.pack<3>("x", vi);
            });

    if (has_input<ZenoBoundary>("ZSBoundary")) {
      auto boundary = get_input<ZenoBoundary>("ZSBoundary");

      using basic_ls_t = typename ZenoLevelSet::basic_ls_t;
      using const_sdf_vel_ls_t = typename ZenoLevelSet::const_sdf_vel_ls_t;
      using const_transition_ls_t =
          typename ZenoLevelSet::const_transition_ls_t;
      if (boundary->zsls)
        match([&](const auto &ls) {
          if constexpr (is_same_v<RM_CVREF_T(ls), basic_ls_t>) {
            match([&](const auto &lsPtr) {
              auto lsv = get_level_set_view<execspace_e::cuda>(lsPtr);
              projectBoundary(cudaPol, lsv, *boundary, verts, dt);
            })(ls._ls);
          } else if constexpr (is_same_v<RM_CVREF_T(ls), const_sdf_vel_ls_t>) {
            match([&](auto lsv) {
              projectBoundary(cudaPol, SdfVelFieldView{lsv}, *boundary, verts,
                              dt);
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
                              *boundary, verts, dt);
            })(ls.template getView<zs::execspace_e::cuda>());
          }
        })(boundary->zsls->getLevelSet());

      if (boundary->type != collider_e::Sticky) {
        cudaPol(Collapse{verts.size()},
                [verts = proxy<execspace_e::cuda>({}, verts)] __device__(
                    int vi) mutable {
                  using mat3 = zs::vec<float, 3, 3>;
                  auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
                  int BCorder = verts("BCorder", vi);
                  if (BCorder != 0) {
                    if (BCorder == 1) {
                      auto n0 = col(BCbasis, 0);
                      auto n1 = n0.orthogonal().normalized();
                      auto n2 = n0.cross(n1).normalized();
                      for (int d = 0; d != 3; ++d) {
                        BCbasis(d, 1) = n1(d);
                        BCbasis(d, 2) = n2(d);
                      }
                    } else if (BCorder == 2) {
                      auto n0 = col(BCbasis, 0);
                      auto n1 = col(BCbasis, 1);
                      auto n2 = n0.cross(n1).normalized();
                      for (int d = 0; d != 3; ++d)
                        BCbasis(d, 2) = n2(d);
                    }
                    verts.tuple<9>("BCbasis", vi) = BCbasis;
                    verts.tuple<3>("BCtarget", vi) =
                        BCbasis.transpose() * verts.pack<3>("BCtarget", vi);
                  }
                });
      }
    }

    set_output("ZSParticles", zsverts);
  }
};

ZENDEFNODE(ApplyBoundaryOnVertices,
           {
               {"ZSParticles", "ZSBoundary", {"float", "dt", "0.01"}},
               {"ZSParticles"},
               {},
               {"FEM"},
           });

struct MoveTowards : INode {
  template <typename VertsT>
  void setupVelocity(zs::CudaExecutionPolicy &pol, VertsT &bouVerts,
                     typename ZenoParticles::particles_t &next, double dt) {
    using namespace zs;
    // update mesh verts
    bouVerts.append_channels(pol, {{"BCtarget", 3}});
    pol(Collapse{bouVerts.size()},
        [prev = proxy<execspace_e::cuda>({}, bouVerts),
         next = proxy<execspace_e::cuda>({}, next),
         dt] __device__(int pi) mutable {
          auto newX = next.pack<3>("x", pi);
          prev.template tuple<3>("BCtarget", pi) = newX;
          prev.template tuple<3>("v", pi) =
              (newX - prev.template pack<3>("x", pi)) / dt;
        });
  }
  void apply() override {
    using namespace zs;
    fmt::print(fg(fmt::color::green), "begin executing MoveTowards\n");

    std::shared_ptr<ZenoParticles> zsprimseq{};

    if (!has_input<ZenoParticles>("ZSParticles"))
      throw std::runtime_error(
          fmt::format("no incoming prim for prim sequence!\n"));
    auto next = get_input<ZenoParticles>("ZSParticles");
    if (!next->asBoundary)
      throw std::runtime_error(
          fmt::format("incoming prim is not used as a boundary!\n"));

    auto cudaPol = cuda_exec().device(0);
    if (has_input<ZenoParticles>("ZSBoundaryPrimitive")) {
      zsprimseq = get_input<ZenoParticles>("ZSBoundaryPrimitive");
      auto numV = zsprimseq->numParticles();
      auto numE = zsprimseq->numElements();
      if (numV != next->numParticles() || numE != next->numElements()) {
        fmt::print("current numVerts {}, numEles ({}).\nIncoming "
                   "boundary primitive numVerts ({}), numEles ({})\n",
                   numV, numE, next->numParticles(), next->numElements());
        throw std::runtime_error(
            fmt::format("prim size mismatch with current sequence prim!\n"));
      }

      auto dt = get_input2<float>("framedt"); // framedt
      /// update velocity
      if (zsprimseq->hasImage(ZenoParticles::s_particleTag))
        setupVelocity(cudaPol, zsprimseq->getParticles<true>(),
                      next->getParticles(), dt);
      setupVelocity(cudaPol, zsprimseq->getParticles(), next->getParticles(),
                    dt);
    }

    fmt::print(fg(fmt::color::cyan), "done executing MoveTowards\n");
    set_output("ZSBoundaryPrimitive", get_input("ZSBoundaryPrimitive"));
  }
};

ZENDEFNODE(MoveTowards, {
                            {"ZSBoundaryPrimitive",
                             {"float", "framedt", "0.1"},
                             "ZSParticles"},
                            {"ZSBoundaryPrimitive"},
                            {},
                            {"FEM"},
                        });

} // namespace zeno