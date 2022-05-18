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
  template <typename LsView>
  constexpr void projectBoundary(zs::CudaExecutionPolicy &cudaPol, LsView lsv,
                                 const ZenoBoundary &boundary,
                                 typename ZenoParticles::particles_t &verts,
                                 float dt) {
    using namespace zs;
    auto collider = boundary.getBoundary(lsv);
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts), boundary = collider,
             dt] __device__(int vi) mutable {
              using mat3 = zs::vec<float, 3, 3>;
              auto vel = verts.pack<3>("v", vi);
              auto pos = verts.pack<3>("x", vi);
              if (boundary.queryInside(pos)) {
                auto v = boundary.getVelocity(pos);
                auto n = boundary.getNormal(pos);
                if (boundary.type == collider_e::Sticky) {
                  verts.tuple<9>("BCbasis", vi) = mat3::identity();
                  verts("BCorder", vi) = reinterpret_bits<float>(3);
                  verts.tuple<3>("BCtarget", vi) =
                      verts.pack<3>("BCtarget", vi) + dt * v;
                } else if (boundary.type == collider_e::Slip) {
                  auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
                  auto BCtarget = verts.pack<3>("BCtarget", vi);
                  auto BCorder = reinterpret_bits<int>(verts("BCorder", vi));
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
                    verts.tuple<9>("BCbasis", vi) = BCbasis;
                    verts("BCorder", vi) = reinterpret_bits<float>(BCorder + 1);
                    verts.tuple<3>("BCtarget", vi) =
                        verts.pack<3>("BCtarget", vi) + dt * v.dot(n) * n;
                  }
                }
              }
            });
  }
  void apply() override {
    auto zsverts = get_input<ZenoParticles>("ZSParticles");
    auto &verts = zsverts->getParticles();
    auto dt = get_input2<float>("dt");

    using namespace zs;

    auto cudaPol = cuda_exec().device(0);

    /// init BC status
    cudaPol(Collapse{verts.size()},
            [verts = proxy<execspace_e::cuda>({}, verts)] __device__(
                int vi) mutable {
              using mat3 = zs::vec<float, 3, 3>;
              verts.tuple<9>("BCbasis", vi) = mat3::identity();
              verts("BCorder", vi) = reinterpret_bits<float>(0);
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
                  auto BCorder = reinterpret_bits<int>(verts("BCorder", vi));
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

} // namespace zeno