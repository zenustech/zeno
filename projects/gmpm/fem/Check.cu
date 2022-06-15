
#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/physics/constitutive_models/StvkWithHencky.hpp"
#include "zensim/types/Property.h"
#include "zensim/types/SmallVector.hpp"
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

/// ref: https://github.com/zfergus/finite-diff
/// ref: ziran
/**
 * https://www.cs.ucr.edu/~craigs/papers/2019-derivatives/course.pdf
 */

struct ChkIpcSystem : INode {
  using T = double;
  using dtiles_t = zs::TileVector<T, 32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;
  using mat3 = zs::vec<T, 3, 3>;
  using pair_t = zs::vec<int, 2>;
  using pair3_t = zs::vec<int, 3>;
  using pair4_t = zs::vec<int, 4>;

  static constexpr bool enable_contact = true;
  static constexpr vec3 s_groundNormal{0, 1, 0};

  inline static T kappa = 1e4;
  inline static T xi = 1e-2; // 2e-3;
  inline static T dHat = 0.004;

  inline static bool enableGravInertia = false;
  inline static bool enableElasticity = true;
  inline static bool enableBarrier = false;
  inline static bool enableGroundBarrier = false;

  static T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
               const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), vertData.size()},
        ret{vertData.get_allocator(), 1};
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<3>(tag0, pi);
              auto v1 = data.pack<3>(tag1, pi);
              res[pi] = v0.dot(v1);
            });
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0);
    return ret.getVal();
  }

  static T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
                   const zs::SmallString tag = "dir") {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), vertData.size()},
        ret{vertData.get_allocator(), 1};
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res),
             tag] __device__(int pi) mutable {
              auto v = data.pack<3>(tag, pi);
              res[pi] = v.abs().max();
            });
    zs::reduce(cudaPol, std::begin(res), std::end(res), std::begin(ret), (T)0,
               getmax<T>{});
    return ret.getVal();
  }

  /// ref: codim-ipc
  static void precompute_constraints(
      zs::CudaExecutionPolicy &pol, ZenoParticles &zstets,
      const dtiles_t &vtemp, T dHat, T xi, zs::Vector<pair_t> &PP,
      zs::Vector<T> &wPP, zs::Vector<int> &nPP, zs::Vector<pair3_t> &PE,
      zs::Vector<T> &wPE, zs::Vector<int> &nPE, zs::Vector<pair4_t> &PT,
      zs::Vector<T> &wPT, zs::Vector<int> &nPT, zs::Vector<pair4_t> &EE,
      zs::Vector<T> &wEE, zs::Vector<int> &nEE, zs::Vector<pair4_t> &PPM,
      zs::Vector<T> &wPPM, zs::Vector<int> &nPPM, zs::Vector<pair4_t> &PEM,
      zs::Vector<T> &wPEM, zs::Vector<int> &nPEM, zs::Vector<pair4_t> &PTM,
      zs::Vector<T> &wPTM, zs::Vector<int> &nPTM, zs::Vector<pair4_t> &EEM,
      zs::Vector<T> &wEEM, zs::Vector<int> &nEEM, const zs::SmallString &xTag) {
    using namespace zs;
    using bv_t = typename ZenoParticles::lbvh_t::Box;
    // dHat = dHat + xi
    T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
    T xi2 = xi * xi;
    constexpr auto space = execspace_e::cuda;
    const auto &verts = zstets.getParticles();
    const auto &eles = zstets.getQuadraturePoints();

    nPP.setVal(0);
    nPE.setVal(0);
    nPT.setVal(0);
    nEE.setVal(0);
    nPPM.setVal(0);
    nPEM.setVal(0);
    nPTM.setVal(0);
    nEEM.setVal(0);

    if (!enableBarrier)
      return;

    /// tri
    const auto &surfaces = zstets[ZenoParticles::s_surfTriTag];
    {
      auto bvs = retrieve_bounding_volumes(pol, vtemp, surfaces, wrapv<3>{},
                                           0.f, xTag);
      if (!zstets.hasBvh(ZenoParticles::s_surfTriTag)) // build if bvh not exist
        zstets.bvh(ZenoParticles::s_surfTriTag).build(pol, bvs);
      else
        zstets.bvh(ZenoParticles::s_surfTriTag).refit(pol, bvs);
    }
    const auto &stBvh = zstets.bvh(ZenoParticles::s_surfTriTag);

    /// edges
    const auto &surfEdges = zstets[ZenoParticles::s_surfEdgeTag];
    {
      auto bvs = retrieve_bounding_volumes(pol, vtemp, surfEdges, wrapv<2>{},
                                           0.f, xTag);
      if (!zstets.hasBvh(ZenoParticles::s_surfEdgeTag))
        zstets.bvh(ZenoParticles::s_surfEdgeTag).build(pol, bvs);
      else
        zstets.bvh(ZenoParticles::s_surfEdgeTag).refit(pol, bvs);
    }
    const auto &seBvh = zstets.bvh(ZenoParticles::s_surfEdgeTag);

    /// points
    const auto &surfVerts = zstets[ZenoParticles::s_surfVertTag];

    // query pt
    pol(Collapse{surfVerts.size()},
        [svs = proxy<space>({}, surfVerts), sts = proxy<space>({}, surfaces),
         verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
         bvh = proxy<space>(stBvh), PP = proxy<space>(PP),
         wPP = proxy<space>(wPP), nPP = proxy<space>(nPP),
         PE = proxy<space>(PE), wPE = proxy<space>(wPE),
         nPE = proxy<space>(nPE), PT = proxy<space>(PT),
         wPT = proxy<space>(wPT), nPT = proxy<space>(nPT), xTag,
         thickness = dHat + xi, xi2,
         activeGap2 = limits<T>::max()] ZS_LAMBDA(int svi) mutable {
          auto vi = reinterpret_bits<int>(svs("inds", svi));
          auto p = vtemp.template pack<3>(xTag, vi);
          auto wp = svs("w", vi) / 4;
          // auto [mi, ma] = get_bounding_box(p - thickness, p + thickness);
          auto [mi, ma] = get_bounding_box(p - 10, p + 10);
          auto bv = bv_t{mi, ma};
          bvh.iter_neighbors(bv, [&](int stI) {
            auto tri = sts.template pack<3>("inds", stI)
                           .template reinterpret_bits<int>();
            if (vi == tri[0] || vi == tri[1] || vi == tri[2])
              return;
            // all affected by sticky boundary conditions
            if ((verts("BCorder", vi)) == 3 &&
                (verts("BCorder", tri[0])) == 3 &&
                (verts("BCorder", tri[1])) == 3 &&
                (verts("BCorder", tri[2])) == 3)
              return;
            // ccd
            auto t0 = vtemp.template pack<3>(xTag, tri[0]);
            auto t1 = vtemp.template pack<3>(xTag, tri[1]);
            auto t2 = vtemp.template pack<3>(xTag, tri[2]);

            switch (pt_distance_type(p, t0, t1, t2)) {
            case 0: {
              if (dist2_pp(p, t0) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{vi, tri[0]};
                wPP[no] = wp;
              }
              break;
            }
            case 1: {
              if (dist2_pp(p, t1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{vi, tri[1]};
                wPP[no] = wp;
              }
              break;
            }
            case 2: {
              if (dist2_pp(p, t2) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{vi, tri[2]};
                wPP[no] = wp;
              }
              break;
            }
            case 3: {
              if (dist2_pe(p, t0, t1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{vi, tri[0], tri[1]};
                wPE[no] = wp;
              }
              break;
            }
            case 4: {
              if (dist2_pe(p, t1, t2) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{vi, tri[1], tri[2]};
                wPE[no] = wp;
              }
              break;
            }
            case 5: {
              if (dist2_pe(p, t2, t0) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{vi, tri[2], tri[0]};
                wPE[no] = wp;
              }
              break;
            }
            case 6: {
              if (dist2_pt(p, t0, t1, t2) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPT[0], 1);
                PT[no] = pair4_t{vi, tri[0], tri[1], tri[2]};
                wPT[no] = wp;
              }
              break;
            }
            default:
              break;
            }
          });
        });

    // query ee
    zs::Vector<T> surfEdgeAlphas{surfEdges.get_allocator(), surfEdges.size()};
    pol(Collapse{surfEdges.size()},
        [ses = proxy<space>({}, surfEdges), verts = proxy<space>({}, verts),
         vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(seBvh),
         PP = proxy<space>(PP), wPP = proxy<space>(wPP),
         nPP = proxy<space>(nPP), PE = proxy<space>(PE),
         wPE = proxy<space>(wPE), nPE = proxy<space>(nPE),
         EE = proxy<space>(EE), wEE = proxy<space>(wEE),
         nEE = proxy<space>(nEE),
         //
         PPM = proxy<space>(PPM), wPPM = proxy<space>(wPPM),
         nPPM = proxy<space>(nPPM), PEM = proxy<space>(PEM),
         wPEM = proxy<space>(wPEM), nPEM = proxy<space>(nPEM),
         EEM = proxy<space>(EEM), wEEM = proxy<space>(wEEM),
         nEEM = proxy<space>(nEEM), xTag, thickness = dHat + xi, xi2,
         activeGap2 = limits<T>::max()] ZS_LAMBDA(int sei) mutable {
          auto eiInds = ses.template pack<2>("inds", sei)
                            .template reinterpret_bits<int>();
          auto selfWe = ses("w", sei);
          bool selfFixed = (verts("BCorder", eiInds[0])) == 3 &&
                           (verts("BCorder", eiInds[1])) == 3;
          auto x0 = vtemp.template pack<3>(xTag, eiInds[0]);
          auto x1 = vtemp.template pack<3>(xTag, eiInds[1]);
          auto ea0Rest = verts.template pack<3>("x0", eiInds[0]);
          auto ea1Rest = verts.template pack<3>("x0", eiInds[1]);
          auto [mi, ma] = get_bounding_box(x0, x1);
          // auto bv = bv_t{mi - thickness, ma + thickness};
          auto bv = bv_t{mi - 10, ma + 10};
          bvh.iter_neighbors(bv, [&](int sej) {
            // if (sei > sej) return;
            auto ejInds = ses.template pack<2>("inds", sej)
                              .template reinterpret_bits<int>();
            if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
                eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
              return;
            // all affected by sticky boundary conditions
            if (selfFixed && (verts("BCorder", ejInds[0])) == 3 &&
                (verts("BCorder", ejInds[1])) == 3)
              return;
            // ccd
            auto eb0 = vtemp.template pack<3>(xTag, ejInds[0]);
            auto eb1 = vtemp.template pack<3>(xTag, ejInds[1]);
            auto eb0Rest = verts.template pack<3>("x0", ejInds[0]);
            auto eb1Rest = verts.template pack<3>("x0", ejInds[1]);

            auto we = (selfWe + ses("w", sej)) / 4;

            // IPC (24)
            T c = (x1 - x0).cross(eb1 - eb0).l2NormSqr();
            T epsX = 1e-3 * (ea0Rest - ea1Rest).l2NormSqr() *
                     (eb0Rest - eb1Rest).l2NormSqr();
            auto cDivEpsX = c / epsX;
            T eem = (2 - cDivEpsX) * cDivEpsX;
            bool mollify = c < epsX;

            switch (ee_distance_type(x0, x1, eb0, eb1)) {
            case 0: {
              if (dist2_pp(x0, eb0) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                  PPM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                  wPPM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{eiInds[0], ejInds[0]};
                  wPP[no] = we;
                }
              }
              break;
            }
            case 1: {
              if (dist2_pp(x0, eb1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                  PPM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[1], ejInds[0]};
                  wPPM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{eiInds[0], ejInds[1]};
                  wPP[no] = we;
                }
              }
              break;
            }
            case 2: {
              if (dist2_pe(x0, eb0, eb1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                  PEM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                  wPEM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{eiInds[0], ejInds[0], ejInds[1]};
                  wPE[no] = we;
                }
              }
              break;
            }
            case 3: {
              if (dist2_pp(x1, eb0) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                  PPM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]};
                  wPPM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{eiInds[1], ejInds[0]};
                  wPP[no] = we;
                }
              }
              break;
            }
            case 4: {
              if (dist2_pp(x1, eb1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPPM[0], 1);
                  PPM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[1], ejInds[0]};
                  wPPM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{eiInds[1], ejInds[1]};
                  wPP[no] = we;
                }
              }
              break;
            }
            case 5: {
              if (dist2_pe(x1, eb0, eb1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                  PEM[no] = pair4_t{eiInds[1], eiInds[0], ejInds[0], ejInds[1]};
                  wPEM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{eiInds[1], ejInds[0], ejInds[1]};
                  wPE[no] = we;
                }
              }
              break;
            }
            case 6: {
              if (dist2_pe(eb0, x0, x1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                  PEM[no] = pair4_t{ejInds[0], ejInds[1], eiInds[0], eiInds[1]};
                  wPEM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{ejInds[0], eiInds[0], eiInds[1]};
                  wPE[no] = we;
                }
              }
              break;
            }
            case 7: {
              if (dist2_pe(eb1, x0, x1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nPEM[0], 1);
                  PEM[no] = pair4_t{ejInds[1], ejInds[0], eiInds[0], eiInds[1]};
                  wPEM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{ejInds[1], eiInds[0], eiInds[1]};
                  wPE[no] = we;
                }
              }
              break;
            }
            case 8: {
              if (dist2_ee(x0, x1, eb0, eb1) - xi2 < activeGap2) {
                if (mollify) {
                  auto no = atomic_add(exec_cuda, &nEEM[0], 1);
                  EEM[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                  wEEM[no] = we;
                } else {
                  auto no = atomic_add(exec_cuda, &nEE[0], 1);
                  EE[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                  wEE[no] = we;
                }
              }
              break;
            }
            default:
              break;
            }
          });
        });
    fmt::print(
        "contact indeed detected. nPP: {}, nPE: {}, nPT: {}, nEE: {}; \n\t"
        "nPPM: {}, nPEM: {}, nPTM: {}, nEEM: {}\n",
        nPP.getVal(), nPE.getVal(), nPT.getVal(), nEE.getVal(), nPPM.getVal(),
        nPEM.getVal(), nPTM.getVal(), nEEM.getVal());
    if (nPPM.getVal() > 0 || nPEM.getVal() > 0 || nPTM.getVal() > 0 ||
        nEEM.getVal() > 0) {
      getchar();
    }
  }

  struct FEMSystem {
    static constexpr auto space = zs::execspace_e::cuda;

    void regen(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      Vector<T> rns(vtemp.size() * 3);
      std::random_device rd{};
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(1, 100);
      for (int i = 0; i != rns.size(); ++i)
        rns[i] = distrib(gen) / (T)100;
      rns = rns.clone({memsrc_e::device, 0});

      pol(Collapse{vtemp.size()},
          [rns = proxy<space>(rns),
           vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
            for (int d = 0; d != 3; ++d)
              vtemp("dx", d, vi) = rns[vi * 3 + d];
          });
    }

    bool check_gradient(zs::CudaExecutionPolicy &pol, const T eps,
                        const T pass_ratio) { // 1e-4, 1e-3
      using namespace zs;
      regen(pol);
      pol(Collapse{vtemp.size()},
          [vtemp = proxy<space>({}, vtemp), eps] __device__(int vi) mutable {
            auto dx = vtemp.template pack<3>("dx", vi) * eps;
            auto x = vtemp.template pack<3>("x", vi);
            vtemp.template tuple<3>("dx", vi) = dx;
            vtemp.template tuple<3>("x0", vi) = x - dx;
            vtemp.template tuple<3>("x1", vi) = x + dx;
          });

      auto dxNorm = std::sqrt(dot(pol, vtemp, "dx", "dx"));
      T f0;
      match([&](auto &elasticModel) { f0 = energy(pol, elasticModel, "x0"); })(
          model);
      prepareGradAndHessian(pol, "x0", "g0", "H0");

      T f1;
      match([&](auto &elasticModel) { f1 = energy(pol, elasticModel, "x1"); })(
          model);
      prepareGradAndHessian(pol, "x1", "g1", "H1");

      //
      double true_value = std::abs(f1 - f0 -
                                   (dot(pol, vtemp, "g0", "dx") +
                                    dot(pol, vtemp, "g1", "dx"))) /
                          dxNorm;
      double fake_value = std::abs(f1 - f0 -
                                   2 * (dot(pol, vtemp, "g0", "dx") +
                                        dot(pol, vtemp, "g1", "dx"))) /
                          dxNorm;
      fmt::print(fg(fmt::color::green),
                 "[check gradient]: realValue: {}, fakeValue: {}, ratio: {}\n",
                 true_value, fake_value, true_value / fake_value);
      {
        pol(Collapse{vtemp.size()},
            [vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
              auto g0 = vtemp.template pack<3>("g0", vi);
              auto g1 = vtemp.template pack<3>("g1", vi);
              vtemp.template tuple<3>("g", vi) = g0 + g1;
            });
        auto gNorm = std::sqrt(dot(pol, vtemp, "g", "g"));
        if (f1 == f0 && gNorm < dxNorm)
          return true;
      }
      return true_value / fake_value < pass_ratio;
    }
    bool check_jacobian(zs::CudaExecutionPolicy &pol, const T eps,
                        const T pass_ratio) { // 1e-4, 1e-3
      using namespace zs;
      return false;
    }
    template <typename Model>
    void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol,
                                          const Model &model,
                                          const zs::SmallString &xTag,
                                          const zs::SmallString &gTag,
                                          const zs::SmallString &hTag) {
      using namespace zs;
      cudaPol(
          zs::range(eles.size()),
          [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, etemp),
           verts = proxy<space>({}, verts), eles = proxy<space>({}, eles),
           model, xTag, gTag, hTag, dt = this->dt] __device__(int ei) mutable {
            auto DmInv = eles.pack<3, 3>("IB", ei);
            auto dFdX = dFdXMatrix(DmInv);
            auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
            vec3 xs[4] = {
                vtemp.pack<3>(xTag, inds[0]), vtemp.pack<3>(xTag, inds[1]),
                vtemp.pack<3>(xTag, inds[2]), vtemp.pack<3>(xTag, inds[3])};
            mat3 F{};
            {
              auto x1x0 = xs[1] - xs[0];
              auto x2x0 = xs[2] - xs[0];
              auto x3x0 = xs[3] - xs[0];
              auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                             x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
              F = Ds * DmInv;
            }
            auto P = model.first_piola(F);
            auto vole = eles("vol", ei);
            auto vecP = flatten(P);
            auto dFdXT = dFdX.transpose();
            auto vfdt2 = -vole * (dFdXT * vecP) * dt * dt;

            for (int i = 0; i != 4; ++i) {
              auto vi = inds[i];
              for (int d = 0; d != 3; ++d)
                atomic_add(exec_cuda, &vtemp(gTag, d, vi), vfdt2(i * 3 + d));
            }

            // hessian rotation: trans^T hess * trans
            // left trans^T: multiplied on rows
            // right trans: multiplied on cols
            mat3 BCbasis[4];
            int BCorder[4];
            for (int i = 0; i != 4; ++i) {
              BCbasis[i] = verts.pack<3, 3>("BCbasis", inds[i]);
              BCorder[i] = (verts("BCorder", inds[i]));
            }
            auto Hq = model.first_piola_derivative(F, false_c);
            auto H = dFdXT * Hq * dFdX * vole * dt * dt;
            // rotate and project
            for (int vi = 0; vi != 4; ++vi) {
              int offsetI = vi * 3;
              for (int vj = 0; vj != 4; ++vj) {
                int offsetJ = vj * 3;
                mat3 tmp{};
                for (int i = 0; i != 3; ++i)
                  for (int j = 0; j != 3; ++j)
                    tmp(i, j) = H(offsetI + i, offsetJ + j);
                // rotate
                tmp = BCbasis[vi].transpose() * tmp * BCbasis[vj];
                // project
                if (BCorder[vi] > 0 || BCorder[vj] > 0) {
                  if (vi == vj) {
                    for (int i = 0; i != BCorder[vi]; ++i)
                      for (int j = 0; j != BCorder[vj]; ++j)
                        tmp(i, j) = (i == j ? 1 : 0);
                  } else {
                    for (int i = 0; i != BCorder[vi]; ++i)
                      for (int j = 0; j != BCorder[vj]; ++j)
                        tmp(i, j) = 0;
                  }
                }
                for (int i = 0; i != 3; ++i)
                  for (int j = 0; j != 3; ++j)
                    H(offsetI + i, offsetJ + j) = tmp(i, j);
              }
            }
            etemp.tuple<12 * 12>(hTag, ei) = H;
          });
    }
    void computeBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol,
                                          const zs::SmallString &xTag,
                                          const zs::SmallString &gTag,
                                          const zs::SmallString &hTag) {
      if (!enableBarrier)
        return;
      using namespace zs;
      T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
      using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
      using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
      using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
      auto numPP = nPP.getVal();
      pol(range(numPP), [vtemp = proxy<space>({}, vtemp),
                         tempPP = proxy<space>({}, tempPP),
                         PP = proxy<space>(PP), wPP = proxy<space>(wPP), xTag,
                         gTag, hTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int ppi) mutable {
        auto pp = PP[ppi];
        auto x0 = vtemp.pack<3>(xTag, pp[0]);
        auto x1 = vtemp.pack<3>(xTag, pp[1]);
        auto ppGrad = dist_grad_pp(x0, x1);
        auto dist2 = dist2_pp(x0, x1);
        if (dist2 < xi2)
          printf("dist already smaller than xi!\n");
        if (dist2 - xi2 < activeGap2) {
          auto grad =
              ppGrad * (-wPP[ppi] * dHat *
                        zs::barrier_gradient(dist2 - xi2, activeGap2, kappa));
          // gradient
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pp[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pp[1]), grad(1, d));
          }
          // hessian
          auto ppHess = dist_hess_pp(x0, x1);
          auto ppGrad_ = Vec6View{ppGrad.data()};
          ppHess =
              wPP[ppi] * dHat *
              (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) *
                   dyadic_prod(ppGrad_, ppGrad_) +
               zs::barrier_gradient(dist2 - xi2, activeGap2, kappa) * ppHess);
          // make pd
          make_pd(ppHess);
          // pp[0], pp[1]
          tempPP.tuple<36>(hTag, ppi) = ppHess;
        } else {
          using Mat6x6 = zs::vec<T, 6, 6>;
          tempPP.tuple<36>(hTag, ppi) = Mat6x6::zeros();
        }
      });
      auto numPE = nPE.getVal();
      pol(range(numPE), [vtemp = proxy<space>({}, vtemp),
                         tempPE = proxy<space>({}, tempPE),
                         PE = proxy<space>(PE), wPE = proxy<space>(wPE), xTag,
                         gTag, hTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int pei) mutable {
        auto pe = PE[pei];
        auto p = vtemp.pack<3>(xTag, pe[0]);
        auto e0 = vtemp.pack<3>(xTag, pe[1]);
        auto e1 = vtemp.pack<3>(xTag, pe[2]);

        auto peGrad = dist_grad_pe(p, e0, e1);
        auto dist2 = dist2_pe(p, e0, e1);
        if (dist2 < xi2)
          printf("dist already smaller than xi!\n");
        if (dist2 - xi2 < activeGap2) {
          auto grad =
              peGrad * (-wPE[pei] * dHat *
                        zs::barrier_gradient(dist2 - xi2, activeGap2, kappa));
          // gradient
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pe[2]), grad(2, d));
          }
          // hessian
          auto peHess = dist_hess_pe(p, e0, e1);
          auto peGrad_ = Vec9View{peGrad.data()};
          peHess =
              wPE[pei] * dHat *
              (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) *
                   dyadic_prod(peGrad_, peGrad_) +
               zs::barrier_gradient(dist2 - xi2, activeGap2, kappa) * peHess);
          // make pd
          make_pd(peHess);
          // pe[0], pe[1], pe[2]
          tempPE.tuple<81>(hTag, pei) = peHess;
        } else {
          using Mat9x9 = zs::vec<T, 9, 9>;
          tempPE.tuple<81>(hTag, pei) = Mat9x9::zeros();
        }
      });
      auto numPT = nPT.getVal();
      pol(range(numPT), [vtemp = proxy<space>({}, vtemp),
                         tempPT = proxy<space>({}, tempPT),
                         PT = proxy<space>(PT), wPT = proxy<space>(wPT), xTag,
                         gTag, hTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int pti) mutable {
        auto pt = PT[pti];
        auto p = vtemp.pack<3>(xTag, pt[0]);
        auto t0 = vtemp.pack<3>(xTag, pt[1]);
        auto t1 = vtemp.pack<3>(xTag, pt[2]);
        auto t2 = vtemp.pack<3>(xTag, pt[3]);

        auto ptGrad = dist_grad_pt(p, t0, t1, t2);
        auto dist2 = dist2_pt(p, t0, t1, t2);
        if (dist2 < xi2)
          printf("dist already smaller than xi!\n");
        if (dist2 - xi2 < activeGap2) {
          auto grad =
              ptGrad * (-wPT[pti] * dHat *
                        zs::barrier_gradient(dist2 - xi2, activeGap2, kappa));
          // gradient
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, pt[3]), grad(3, d));
          }
          // hessian
          auto ptHess = dist_hess_pt(p, t0, t1, t2);
          auto ptGrad_ = Vec12View{ptGrad.data()};
          ptHess =
              wPT[pti] * dHat *
              (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) *
                   dyadic_prod(ptGrad_, ptGrad_) +
               zs::barrier_gradient(dist2 - xi2, activeGap2, kappa) * ptHess);
          // make pd
          make_pd(ptHess);
          // pt[0], pt[1], pt[2], pt[3]
          tempPT.tuple<144>(hTag, pti) = ptHess;
        } else {
          using Mat12x12 = zs::vec<T, 12, 12>;
          tempPT.tuple<144>(hTag, pti) = Mat12x12::zeros();
        }
      });
      auto numEE = nEE.getVal();
      pol(range(numEE), [vtemp = proxy<space>({}, vtemp),
                         tempEE = proxy<space>({}, tempEE),
                         EE = proxy<space>(EE), wEE = proxy<space>(wEE), xTag,
                         gTag, hTag, xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int eei) mutable {
        auto ee = EE[eei];
        auto ea0 = vtemp.pack<3>(xTag, ee[0]);
        auto ea1 = vtemp.pack<3>(xTag, ee[1]);
        auto eb0 = vtemp.pack<3>(xTag, ee[2]);
        auto eb1 = vtemp.pack<3>(xTag, ee[3]);

        auto eeGrad = dist_grad_ee(ea0, ea1, eb0, eb1);
        auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
        if (dist2 < xi2)
          printf("dist already smaller than xi!\n");
        if (dist2 - xi2 < activeGap2) {
          auto grad =
              eeGrad * (-wEE[eei] * dHat *
                        zs::barrier_gradient(dist2 - xi2, activeGap2, kappa));
          // gradient
          for (int d = 0; d != 3; ++d) {
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp(gTag, d, ee[3]), grad(3, d));
          }
          // hessian
          auto eeHess = dist_hess_ee(ea0, ea1, eb0, eb1);
          auto eeGrad_ = Vec12View{eeGrad.data()};
          eeHess =
              wEE[eei] * dHat *
              (zs::barrier_hessian(dist2 - xi2, activeGap2, kappa) *
                   dyadic_prod(eeGrad_, eeGrad_) +
               zs::barrier_gradient(dist2 - xi2, activeGap2, kappa) * eeHess);
          // make pd
          make_pd(eeHess);
          // ee[0], ee[1], ee[2], ee[3]
          tempEE.tuple<144>(hTag, eei) = eeHess;
        } else {
          using Mat12x12 = zs::vec<T, 12, 12>;
          tempEE.tuple<144>(hTag, eei) = Mat12x12::zeros();
        }
      });
      return;
    }
    void computeBoundaryBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol,
                                                  const zs::SmallString &xTag,
                                                  const zs::SmallString &gTag,
                                                  const zs::SmallString &hTag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp), tempPB = proxy<space>({}, tempPB),
           xTag, gTag, hTag, gn = s_groundNormal, dHat2 = dHat * dHat,
           kappa = kappa] ZS_LAMBDA(int vi) mutable {
            auto x = vtemp.pack<3>(xTag, vi);
            auto dist = gn.dot(x);
            auto dist2 = dist * dist;
            auto t = dist2 - dHat2;
            auto g_b = t * zs::log(dist2 / dHat2) * -2 - (t * t) / dist2;
            auto H_b = (zs::log(dist2 / dHat2) * -2.0 - t * 4.0 / dist2) +
                       1.0 / (dist2 * dist2) * (t * t);
            if (dist2 < dHat2) {
              auto grad = -gn * (kappa * g_b * 2 * dist);
              for (int d = 0; d != 3; ++d)
                atomic_add(exec_cuda, &vtemp(gTag, d, vi), grad(d));
            }

            auto param = 4 * H_b * dist2 + 2 * g_b;
            auto hess = mat3::zeros();
            if (dist2 < dHat2 && param > 0) {
              auto nn = dyadic_prod(gn, gn);
              hess = (kappa * param) * nn;
            }
            tempPB.tuple<9>(hTag, vi) = hess;
          });
      return;
    }

    void prepareGradAndHessian(zs::CudaExecutionPolicy &pol,
                               const zs::SmallString &xTag,
                               const zs::SmallString &gTag,
                               const zs::SmallString &hTag) {
      using namespace zs;
      pol(zs::range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp), gTag] __device__(int i) mutable {
            vtemp.template tuple<3>(gTag, i) = zs::vec<T, 3>::zeros();
          });
      if (enableGravInertia) {
        pol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt = this->dt, xTag, gTag] __device__(int i) mutable {
              auto m = verts("m", i);
              auto v = verts.pack<3>("v", i);
              vtemp.template tuple<3>(gTag, i) =
                  vtemp.template pack<3>(gTag, i) +
                  m * vec3{0, -9, 0} * dt * dt -
                  m * (vtemp.pack<3>(xTag, i) - vtemp.pack<3>("xtilde", i));
            });
      }
      if (enableElasticity) {
        match([&](auto &elasticModel) {
          computeElasticGradientAndHessian(pol, elasticModel, xTag, gTag, hTag);
        })(model);
      }
      if (enableBarrier) {
        precompute_constraints(pol, zstets, vtemp, dHat, xi, PP, wPP, nPP, PE,
                               wPE, nPE, PT, wPT, nPT, EE, wEE, nEE, PPM, wPPM,
                               nPPM, PEM, wPEM, nPEM, PTM, wPTM, nPTM, EEM,
                               wEEM, nEEM, xTag);
        computeBarrierGradientAndHessian(pol, xTag, gTag, hTag);
      }
      if (enableGroundBarrier) {
        computeBoundaryBarrierGradientAndHessian(pol, xTag, gTag, hTag);
      }
    }

    template <typename Pol, typename Model>
    T energy(Pol &pol, const Model &model, const zs::SmallString tag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
      if (enableGravInertia) {
        pol(range(vtemp.size()),
            [verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
             res = proxy<space>(res), tag,
             dt = this->dt] __device__(int vi) mutable {
              // inertia
              auto m = verts("m", vi);
              auto x = vtemp.pack<3>(tag, vi);
              atomic_add(exec_cuda, &res[0],
                         (T)0.5 * m *
                             (x - vtemp.pack<3>("xtilde", vi)).l2NormSqr());
              // gravity
              atomic_add(exec_cuda, &res[0],
                         -m * vec3{0, -9, 0}.dot(x - verts.pack<3>("x", vi)) *
                             dt * dt);
            });
      }
      // elasticity
      if (enableElasticity) {
        pol(range(eles.size()),
            [verts = proxy<space>({}, verts), eles = proxy<space>({}, eles),
             vtemp = proxy<space>({}, vtemp), res = proxy<space>(res), tag,
             model = model, dt = this->dt] __device__(int ei) mutable {
              auto DmInv = eles.template pack<3, 3>("IB", ei);
              auto inds = eles.template pack<4>("inds", ei)
                              .template reinterpret_bits<int>();
              vec3 xs[4] = {vtemp.template pack<3>(tag, inds[0]),
                            vtemp.template pack<3>(tag, inds[1]),
                            vtemp.template pack<3>(tag, inds[2]),
                            vtemp.template pack<3>(tag, inds[3])};
              mat3 F{};
              {
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                               x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                F = Ds * DmInv;
              }
              auto psi = model.psi(F);
              auto vole = eles("vol", ei);
              atomic_add(exec_cuda, &res[0], vole * psi * dt * dt);
            });
      }
      // contacts
      if (enableBarrier) {
        precompute_constraints(pol, zstets, vtemp, dHat, xi, PP, wPP, nPP, PE,
                               wPE, nPE, PT, wPT, nPT, EE, wEE, nEE, PPM, wPPM,
                               nPPM, PEM, wPEM, nPEM, PTM, wPTM, nPTM, EEM,
                               wEEM, nEEM, tag);
        auto activeGap2 = dHat * dHat + 2 * xi * dHat;
        auto numPP = nPP.getVal();
        pol(range(numPP),
            [vtemp = proxy<space>({}, vtemp), tempPP = proxy<space>({}, tempPP),
             PP = proxy<space>(PP), wPP = proxy<space>(wPP),
             res = proxy<space>(res), tag, xi2 = xi * xi, dHat = dHat,
             activeGap2, kappa = kappa] __device__(int ppi) mutable {
              auto pp = PP[ppi];
              auto x0 = vtemp.pack<3>(tag, pp[0]);
              auto x1 = vtemp.pack<3>(tag, pp[1]);
              auto dist2 = dist2_pp(x0, x1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              if (dist2 - xi2 < activeGap2)
                atomic_add(exec_cuda, &res[0],
                           wPP[ppi] * dHat *
                               zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numPE = nPE.getVal();
        pol(range(numPE),
            [vtemp = proxy<space>({}, vtemp), tempPE = proxy<space>({}, tempPE),
             PE = proxy<space>(PE), wPE = proxy<space>(wPE),
             res = proxy<space>(res), tag, xi2 = xi * xi, dHat = dHat,
             activeGap2, kappa = kappa] __device__(int pei) mutable {
              auto pe = PE[pei];
              auto p = vtemp.pack<3>(tag, pe[0]);
              auto e0 = vtemp.pack<3>(tag, pe[1]);
              auto e1 = vtemp.pack<3>(tag, pe[2]);

              auto dist2 = dist2_pe(p, e0, e1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              if (dist2 - xi2 < activeGap2)
                atomic_add(exec_cuda, &res[0],
                           wPE[pei] * dHat *
                               zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numPT = nPT.getVal();
        pol(range(numPT),
            [vtemp = proxy<space>({}, vtemp), tempPT = proxy<space>({}, tempPT),
             PT = proxy<space>(PT), wPT = proxy<space>(wPT),
             res = proxy<space>(res), tag, xi2 = xi * xi, dHat = dHat,
             activeGap2, kappa = kappa] __device__(int pti) mutable {
              auto pt = PT[pti];
              auto p = vtemp.pack<3>(tag, pt[0]);
              auto t0 = vtemp.pack<3>(tag, pt[1]);
              auto t1 = vtemp.pack<3>(tag, pt[2]);
              auto t2 = vtemp.pack<3>(tag, pt[3]);

              auto dist2 = dist2_pt(p, t0, t1, t2);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              if (dist2 - xi2 < activeGap2)
                atomic_add(exec_cuda, &res[0],
                           wPT[pti] * dHat *
                               zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numEE = nEE.getVal();
        pol(range(numEE),
            [vtemp = proxy<space>({}, vtemp), tempEE = proxy<space>({}, tempEE),
             EE = proxy<space>(EE), wEE = proxy<space>(wEE),
             res = proxy<space>(res), tag, xi2 = xi * xi, dHat = dHat,
             activeGap2, kappa = kappa] __device__(int eei) mutable {
              auto ee = EE[eei];
              auto ea0 = vtemp.pack<3>(tag, ee[0]);
              auto ea1 = vtemp.pack<3>(tag, ee[1]);
              auto eb0 = vtemp.pack<3>(tag, ee[2]);
              auto eb1 = vtemp.pack<3>(tag, ee[3]);

              auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              if (dist2 - xi2 < activeGap2)
                atomic_add(exec_cuda, &res[0],
                           wEE[eei] * dHat *
                               zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
      }
      if (enableGroundBarrier) {
        // boundary
        pol(range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), res = proxy<space>(res), tag,
             gn = s_groundNormal, dHat2 = dHat * dHat,
             kappa = kappa] ZS_LAMBDA(int vi) mutable {
              auto x = vtemp.pack<3>(tag, vi);
              auto dist = gn.dot(x);
              auto dist2 = dist * dist;
              if (dist2 < dHat2) {
                auto temp = -(dist2 - dHat2) * (dist2 - dHat2) *
                            zs::log(dist2 / dHat2) * kappa;
                atomic_add(exec_cuda, &res[0], temp);
              }
            });
      }
      return res.getVal();
    }

    template <typename Pol>
    void multiply(Pol &pol, const zs::SmallString dxTag,
                  const zs::SmallString hTag, const zs::SmallString bTag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      constexpr auto execTag = wrapv<space>{};
      const auto numVerts = verts.size();
      const auto numEles = eles.size();
      // hessian rotation: trans^T hess * trans
      // left trans^T: multiplied on rows
      // right trans: multiplied on cols
      // dx -> b
      pol(range(numVerts), [execTag, vtemp = proxy<space>({}, vtemp),
                            bTag] ZS_LAMBDA(int vi) mutable {
        vtemp.template tuple<3>(bTag, vi) = vec3::zeros();
      });
      // inertial
      if (enableGravInertia) {
        pol(range(numVerts), [execTag, verts = proxy<space>({}, verts),
                              vtemp = proxy<space>({}, vtemp), dxTag,
                              bTag] ZS_LAMBDA(int vi) mutable {
          auto m = verts("m", vi);
          auto dx = vtemp.template pack<3>(dxTag, vi);
          auto BCbasis = verts.template pack<3, 3>("BCbasis", vi);
          dx = BCbasis.transpose() * m * BCbasis * dx;
          for (int d = 0; d != 3; ++d)
            atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
        });
      }
      // elasticity
      if (enableElasticity) {
        pol(range(numEles),
            [execTag, etemp = proxy<space>({}, etemp),
             vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles),
             dxTag, hTag, bTag] ZS_LAMBDA(int ei) mutable {
              constexpr int dim = 3;
              constexpr auto dimp1 = dim + 1;
              auto inds = eles.template pack<dimp1>("inds", ei)
                              .template reinterpret_bits<int>();
              zs::vec<T, dimp1 * dim> temp{};
              for (int vi = 0; vi != dimp1; ++vi)
                for (int d = 0; d != dim; ++d) {
                  temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
                }
              auto He = etemp.template pack<dim * dimp1, dim * dimp1>(hTag, ei);

              temp = He * temp;

              for (int vi = 0; vi != dimp1; ++vi)
                for (int d = 0; d != dim; ++d) {
                  atomic_add(execTag, &vtemp(bTag, d, inds[vi]),
                             temp[vi * dim + d]);
                }
            });
      }
      // contacts
      if (enableBarrier) {
        auto numPP = nPP.getVal();
        pol(range(numPP), [execTag, tempPP = proxy<space>({}, tempPP),
                           vtemp = proxy<space>({}, vtemp), dxTag, hTag, bTag,
                           PP = proxy<space>(PP)] ZS_LAMBDA(int ppi) mutable {
          constexpr int dim = 3;
          auto pp = PP[ppi];
          zs::vec<T, dim * 2> temp{};
          for (int vi = 0; vi != 2; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pp[vi]);
            }
          auto ppHess = tempPP.template pack<6, 6>(hTag, ppi);

          temp = ppHess * temp;

          for (int vi = 0; vi != 2; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pp[vi]), temp[vi * dim + d]);
            }
        });
        auto numPE = nPE.getVal();
        pol(range(numPE), [execTag, tempPE = proxy<space>({}, tempPE),
                           vtemp = proxy<space>({}, vtemp), dxTag, hTag, bTag,
                           PE = proxy<space>(PE)] ZS_LAMBDA(int pei) mutable {
          constexpr int dim = 3;
          auto pe = PE[pei];
          zs::vec<T, dim * 3> temp{};
          for (int vi = 0; vi != 3; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pe[vi]);
            }
          auto peHess = tempPE.template pack<9, 9>(hTag, pei);

          temp = peHess * temp;

          for (int vi = 0; vi != 3; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pe[vi]), temp[vi * dim + d]);
            }
        });
        auto numPT = nPT.getVal();
        pol(range(numPT), [execTag, tempPT = proxy<space>({}, tempPT),
                           vtemp = proxy<space>({}, vtemp), dxTag, hTag, bTag,
                           PT = proxy<space>(PT)] ZS_LAMBDA(int pti) mutable {
          constexpr int dim = 3;
          auto pt = PT[pti];
          zs::vec<T, dim * 4> temp{};
          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pt[vi]);
            }
          auto ptHess = tempPT.template pack<12, 12>(hTag, pti);

          temp = ptHess * temp;

          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pt[vi]), temp[vi * dim + d]);
            }
        });
        auto numEE = nEE.getVal();
        pol(range(numEE), [execTag, tempEE = proxy<space>({}, tempEE),
                           vtemp = proxy<space>({}, vtemp), dxTag, hTag, bTag,
                           EE = proxy<space>(EE)] ZS_LAMBDA(int eei) mutable {
          constexpr int dim = 3;
          auto ee = EE[eei];
          zs::vec<T, dim * 4> temp{};
          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, ee[vi]);
            }
          auto eeHess = tempEE.template pack<12, 12>(hTag, eei);

          temp = eeHess * temp;

          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, ee[vi]), temp[vi * dim + d]);
            }
        });
      } // end contacts
      {
        // boundary
        pol(range(verts.size()), [execTag, vtemp = proxy<space>({}, vtemp),
                                  tempPB = proxy<space>({}, tempPB), dxTag,
                                  hTag, bTag] ZS_LAMBDA(int vi) mutable {
          auto dx = vtemp.template pack<3>(dxTag, vi);
          auto pbHess = tempPB.template pack<3, 3>(hTag, vi);
          dx = pbHess * dx;
          for (int d = 0; d != 3; ++d)
            atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
        });
      }
    }

    FEMSystem(ZenoParticles &zstets, const tiles_t &verts, const tiles_t &eles,
              const ElasticModel &model, dtiles_t &vtemp, dtiles_t &etemp,
              zs::Vector<pair_t> &PP, zs::Vector<T> &wPP, zs::Vector<int> &nPP,
              zs::Vector<pair3_t> &PE, zs::Vector<T> &wPE, zs::Vector<int> &nPE,
              zs::Vector<pair4_t> &PT, zs::Vector<T> &wPT, zs::Vector<int> &nPT,
              zs::Vector<pair4_t> &EE, zs::Vector<T> &wEE, zs::Vector<int> &nEE,
              // mollified
              zs::Vector<pair4_t> &PPM, zs::Vector<T> &wPPM,
              zs::Vector<int> &nPPM, zs::Vector<pair4_t> &PEM,
              zs::Vector<T> &wPEM, zs::Vector<int> &nPEM,
              zs::Vector<pair4_t> &PTM, zs::Vector<T> &wPTM,
              zs::Vector<int> &nPTM, zs::Vector<pair4_t> &EEM,
              zs::Vector<T> &wEEM, zs::Vector<int> &nEEM, T dHat, T xi, T dt)
        : zstets{zstets}, verts{verts}, eles{eles}, model{model}, vtemp{vtemp},
          etemp{etemp}, PP{PP}, wPP{wPP}, nPP{nPP}, tempPP{PP.get_allocator(),
                                                           {{"H0", 36},
                                                            {"H1", 36}},
                                                           PP.size()},
          PE{PE}, wPE{wPE}, nPE{nPE}, tempPE{PE.get_allocator(),
                                             {{"H0", 81}, {"H1", 81}},
                                             PE.size()},
          PT{PT}, wPT{wPT}, nPT{nPT}, tempPT{PT.get_allocator(),
                                             {{"H0", 144}, {"H1", 144}},
                                             PT.size()},
          EE{EE}, wEE{wEE}, nEE{nEE}, tempEE{EE.get_allocator(),
                                             {{"H0", 144}, {"H1", 144}},
                                             EE.size()},
          // mollified
          PPM{PPM}, wPPM{wPPM}, nPPM{nPPM}, tempPPM{PPM.get_allocator(),
                                                    {{"H0", 144}, {"H1", 144}},
                                                    PPM.size()},
          PEM{PEM}, wPEM{wPEM}, nPEM{nPEM}, tempPEM{PEM.get_allocator(),
                                                    {{"H0", 144}, {"H1", 144}},
                                                    PEM.size()},
          PTM{PTM}, wPTM{wPTM}, nPTM{nPTM}, tempPTM{PTM.get_allocator(),
                                                    {{"H0", 144}, {"H1", 144}},
                                                    PTM.size()},
          EEM{EEM}, wEEM{wEEM}, nEEM{nEEM}, tempEEM{EEM.get_allocator(),
                                                    {{"H0", 144}, {"H1", 144}},
                                                    EEM.size()},
          tempPB{verts.get_allocator(), {{"H0", 9}, {"H1", 9}}, verts.size()},
          dHat{dHat}, xi{xi}, dt{dt} {}

    ZenoParticles &zstets;
    const tiles_t &verts;
    const tiles_t &eles;
    const ElasticModel &model;
    dtiles_t &vtemp;
    dtiles_t &etemp;
    // contacts
    zs::Vector<pair_t> &PP;
    zs::Vector<T> &wPP;
    zs::Vector<int> &nPP;
    dtiles_t tempPP;
    zs::Vector<pair3_t> &PE;
    zs::Vector<T> &wPE;
    zs::Vector<int> &nPE;
    dtiles_t tempPE;
    zs::Vector<pair4_t> &PT;
    zs::Vector<T> &wPT;
    zs::Vector<int> &nPT;
    dtiles_t tempPT;
    zs::Vector<pair4_t> &EE;
    zs::Vector<T> &wEE;
    zs::Vector<int> &nEE;
    dtiles_t tempEE;
    // mollified
    zs::Vector<pair4_t> &PPM;
    zs::Vector<T> &wPPM;
    zs::Vector<int> &nPPM;
    dtiles_t tempPPM;
    zs::Vector<pair4_t> &PEM;
    zs::Vector<T> &wPEM;
    zs::Vector<int> &nPEM;
    dtiles_t tempPEM;
    zs::Vector<pair4_t> &PTM;
    zs::Vector<T> &wPTM;
    zs::Vector<int> &nPTM;
    dtiles_t tempPTM;
    zs::Vector<pair4_t> &EEM;
    zs::Vector<T> &wEEM;
    zs::Vector<int> &nEEM;
    dtiles_t tempEEM;

    // boundary contacts
    dtiles_t tempPB;
    // end contacts
    T dHat, xi, dt;
  };

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    std::shared_ptr<ZenoParticles> zsboundary;
    if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
      zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");

    auto activateGravInertial = get_param<int>("grav_inertia");
    auto activateElasticity = get_param<int>("elasticity");
    auto activateBarrier = get_param<int>("barrier");
    auto activateGroundBarrier = get_param<int>("ground_barrier");

    enableGravInertia = activateGravInertial ? 1 : 0;
    enableElasticity = activateElasticity ? 1 : 0;
    enableBarrier = activateBarrier ? 1 : 0;
    enableGroundBarrier = activateGroundBarrier ? 1 : 0;

    auto &verts = zstets->getParticles();
    auto &eles = zstets->getQuadraturePoints();

    static dtiles_t vtemp{verts.get_allocator(),
                          {{"dx", 3},
                           {"xtilde", 3},
                           {"x", 3},
                           {"x0", 3},
                           {"x1", 3},
                           {"g", 3},
                           {"g0", 3},
                           {"g1", 3}},
                          verts.size()};
    static dtiles_t etemp{
        eles.get_allocator(), {{"H0", 12 * 12}, {"H1", 12 * 12}}, eles.size()};
    static Vector<pair_t> PP{verts.get_allocator(), 100000};
    static Vector<T> wPP{verts.get_allocator(), 100000};
    static Vector<int> nPP{verts.get_allocator(), 1};
    static Vector<pair3_t> PE{verts.get_allocator(), 100000};
    static Vector<T> wPE{verts.get_allocator(), 100000};
    static Vector<int> nPE{verts.get_allocator(), 1};
    static Vector<pair4_t> PT{verts.get_allocator(), 100000};
    static Vector<T> wPT{verts.get_allocator(), 100000};
    static Vector<int> nPT{verts.get_allocator(), 1};
    static Vector<pair4_t> EE{verts.get_allocator(), 100000};
    static Vector<T> wEE{verts.get_allocator(), 100000};
    static Vector<int> nEE{verts.get_allocator(), 1};
    // mollified
    static Vector<pair4_t> PPM{verts.get_allocator(), 50000};
    static Vector<T> wPPM{verts.get_allocator(), 50000};
    static Vector<int> nPPM{verts.get_allocator(), 1};
    static Vector<pair4_t> PEM{verts.get_allocator(), 50000};
    static Vector<T> wPEM{verts.get_allocator(), 50000};
    static Vector<int> nPEM{verts.get_allocator(), 1};
    static Vector<pair4_t> PTM{verts.get_allocator(), 50000};
    static Vector<T> wPTM{verts.get_allocator(), 50000};
    static Vector<int> nPTM{verts.get_allocator(), 1};
    static Vector<pair4_t> EEM{verts.get_allocator(), 50000};
    static Vector<T> wEEM{verts.get_allocator(), 50000};
    static Vector<int> nEEM{verts.get_allocator(), 1};

    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{*zstets, verts, eles, models.getElasticModel(),
                vtemp,   etemp, PP,   wPP,
                nPP,     PE,    wPE,  nPE,
                PT,      wPT,   nPT,  EE,
                wEE,     nEE,   PPM,  wPPM,
                nPPM,    PEM,   wPEM, nPEM,
                PTM,     wPTM,  nPTM, EEM,
                wEEM,    nEEM,  dHat, xi,
                dt};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    /// time integrator
    // predict pos
    cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt] __device__(int i) mutable {
              auto x = verts.pack<3>("x", i);
              auto v = verts.pack<3>("v", i);
              vtemp.template tuple<3>("xtilde", i) = x + v * dt;
            });
    // fix initial x for all bcs if not feasible
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp),
             verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto x = verts.pack<3>("x", vi);
              if (int BCorder = (verts("BCorder", vi)); BCorder > 0) {
                auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
                auto BCtarget = verts.pack<3>("BCtarget", vi);
                x = BCbasis.transpose() * x;
                for (int d = 0; d != BCorder; ++d)
                  x[d] = BCtarget[d];
                x = BCbasis * x;
                verts.tuple<3>("x", vi) = x;
              }
              vtemp.tuple<3>("x", vi) = x;
            });

    /// optimizer
    fmt::print("begin checking\n");

    // unit testings
    using vec1 = zs::vec<T, 1>;
    using mat1 = zs::vec<T, 1, 1>;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;
    using vec6 = zs::vec<T, 6>;
    using mat6 = zs::vec<T, 6, 6>;
    using vec9 = zs::vec<T, 9>;
    using mat9 = zs::vec<T, 9, 9>;
    using vec12 = zs::vec<T, 12>;
    using mat12 = zs::vec<T, 12, 12>;
    using Vec3View = zs::vec_view<T, zs::integer_seq<int, 3>>;
    using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
    using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
    using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;

    // barrier
    auto f = [](T *x) { return barrier(*x, 1.23123, 4.1212); };
    auto g = [](T *x) {
      return vec1{zs::barrier_gradient(*x, 1.23123, 4.1212)};
    };
    auto h = [](T *x) {
      return mat1{zs::barrier_hessian(*x, 1.23123, 4.1212)};
    };

    // pp
    auto f_pp = [](T *x_) {
      Vec3View p0{x_}, p1{x_ + 3};
      return dist2_pp(p0, p1);
    };
    auto g_pp = [](T *x_) {
      Vec3View p0{x_}, p1{x_ + 3};
      auto tmp = dist_grad_pp(p0, p1);
      vec6 res{};
      for (int i = 0; i < 6; ++i)
        res.val(i) = tmp.val(i);
      return res;
    };
    auto h_pp = [](T *x_) {
      Vec3View p0{x_}, p1{x_ + 3};
      return dist_hess_pp(p0, p1);
    };

    // pe
    auto f_pe = [](T *x_) {
      Vec3View p{x_}, e0{x_ + 3}, e1{x_ + 6};
      return dist2_pe(p, e0, e1);
    };
    auto g_pe = [](T *x_) {
      Vec3View p{x_}, e0{x_ + 3}, e1{x_ + 6};
      auto tmp = dist_grad_pe(p, e0, e1);
      vec9 res{};
      for (int i = 0; i < 9; ++i)
        res.val(i) = tmp.val(i);
      return res;
    };
    auto h_pe = [](T *x_) {
      Vec3View p{x_}, e0{x_ + 3}, e1{x_ + 6};
      return dist_hess_pe(p, e0, e1);
    };

    // pt
    auto f_pt = [](T *x_) {
      Vec3View p{x_}, t0{x_ + 3}, t1{x_ + 6}, t2{x_ + 9};
      return dist2_pt(p, t0, t1, t2);
    };
    auto g_pt = [](T *x_) {
      Vec3View p{x_}, t0{x_ + 3}, t1{x_ + 6}, t2{x_ + 9};
      auto tmp = dist_grad_pt(p, t0, t1, t2);
      vec12 res{};
      for (int i = 0; i < 12; ++i)
        res.val(i) = tmp.val(i);
      return res;
    };
    auto h_pt = [](T *x_) {
      Vec3View p{x_}, t0{x_ + 3}, t1{x_ + 6}, t2{x_ + 9};
      return dist_hess_pt(p, t0, t1, t2);
    };

    // ee
    auto f_ee = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      return dist2_ee(a, b, c, d);
    };
    auto g_ee = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      auto tmp = dist_grad_ee(a, b, c, d);
      vec12 res{};
      for (int i = 0; i < 12; ++i)
        res.val(i) = tmp.val(i);
      return res;
    };
    auto h_ee = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      return dist_hess_ee(a, b, c, d);
    };

    // ee mollifier
    auto f_eem = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      return mollifier_ee(a, b, c, d, 1);
    };
    auto g_eem = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      auto tmp = mollifier_grad_ee(a, b, c, d, 1);
      vec12 res{};
      for (int i = 0; i < 12; ++i)
        res.val(i) = tmp.val(i);
      return res;
    };
    auto h_eem = [](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      return mollifier_hess_ee(a, b, c, d, 1);
    };

    // elasticity
    // zs::StvkWithHencky<T> model{10000, 0.4};
    zs::FixedCorotated<T> model{10000, 0.3};
    vec3 X0, X1, X2, X3;
    mat3 DmInv{};
    T vol{};
    {
      auto hverts = verts.clone({memsrc_e::host, -1});
      auto hview = proxy<zs::execspace_e::host>({}, hverts);
      X0 = hview.pack<3>("x", 0);
      X1 = hview.pack<3>("x", 1);
      X2 = hview.pack<3>("x", 2);
      X3 = hview.pack<3>("x", 3);

      vec3 ds[3] = {X1 - X0, X2 - X0, X3 - X0};
      mat3 D{};
      for (int d = 0; d != 3; ++d)
        for (int i = 0; i != 3; ++i)
          D(d, i) = ds[i][d];
      vol = std::abs(determinant(D)) / 6;
      DmInv = inverse(D); // safe to call outside kernel
    }
    auto f_e = [&](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
#if 0
      auto x0 = X0 + a;
      auto x1 = X1 + b;
      auto x2 = X2 + c;
      auto x3 = X3 + d;
#else
      auto x0 = a.clone();
      auto x1 = b.clone();
      auto x2 = c.clone();
      auto x3 = d.clone();
#endif
      mat3 F{};
      {
        auto x1x0 = x1 - x0;
        auto x2x0 = x2 - x0;
        auto x3x0 = x3 - x0;
        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                       x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
        F = Ds * DmInv;
      }
      return model.psi(F) * vol;
    };
    auto g_e = [&](T *x_) {
      Vec3View a{x_}, b{x_ + 3}, c{x_ + 6}, d{x_ + 9};
      auto x0 = a.clone();
      auto x1 = b.clone();
      auto x2 = c.clone();
      auto x3 = d.clone();
      mat3 F{};
      {
        auto x1x0 = x1 - x0;
        auto x2x0 = x2 - x0;
        auto x3x0 = x3 - x0;
        auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                       x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
        F = Ds * DmInv;
      }
      return flatten(model.first_piola(F)) * vol;
    };
    auto h_e = [&model](T *x_) {};

    auto checkGradEle = [&](auto &&f, auto &&g, auto v_c, T ep = 1e-4) -> bool {
      constexpr auto passrate = (T)1e-3;

      constexpr int dim = RM_CVREF_T(v_c)::value;
      using VecT = zs::vec<T, dim>;
      VecT x{}, dx{};

      std::random_device rd{};
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(1, 100);
      // check gradient
      for (auto &&[v, dv] : zip(x._data, dx._data)) {
        v = distrib(gen) / (T)100;
        dv = distrib(gen) / (T)100 * ep;
      }

      auto dxNorm = dx.norm();
      auto x0 = x - dx;
      auto x1 = x + dx;
      auto f0 = f(x0.data());
      auto f1 = f(x1.data());
      auto g0 = g(x0.data());
      auto g1 = g(x1.data());
      double true_value = std::abs(f1 - f0 - (g0 + g1).dot(dx)) / dxNorm;
      double fake_value = std::abs(f1 - f0 - 2 * (g0 + g1).dot(dx)) / dxNorm;
      fmt::print(fg(fmt::color::green),
                 "[check gradient]: realValue: {}, fakeValue: {}, ratio: {}\n",
                 true_value, fake_value, true_value / fake_value);
      bool res = false;
      if (f1 == f0 && (g1 + g0).norm() < dxNorm)
        res = true;
      if (true_value / fake_value < passrate)
        res = true;
      fmt::print("\tgradient result: {} (dxNorm: {})\n", res, dxNorm);
      return res;
    };
    auto checkGrad = [&](auto &&f, auto &&g, auto v_c, T ep = 1e-4) -> bool {
      constexpr auto passrate = (T)1e-3;

      constexpr int dim = RM_CVREF_T(v_c)::value;
      using VecT = zs::vec<T, dim>;
      VecT x{}, dx{};

      std::random_device rd{};
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(1, 100);
      // check gradient
      for (auto &&[v, dv] : zip(x._data, dx._data)) {
        v = distrib(gen) / (T)100;
        dv = distrib(gen) / (T)100 * ep;
      }

      auto dxNorm = dx.norm();
      auto x0 = x - dx;
      auto x1 = x + dx;
      auto f0 = f(x0.data());
      auto f1 = f(x1.data());
      auto g0 = g(x0.data());
      auto g1 = g(x1.data());
      double true_value = std::abs(f1 - f0 - (g0 + g1).dot(dx)) / dxNorm;
      double fake_value = std::abs(f1 - f0 - 2 * (g0 + g1).dot(dx)) / dxNorm;
      fmt::print(fg(fmt::color::green),
                 "[check gradient]: realValue: {}, fakeValue: {}, ratio: {}\n",
                 true_value, fake_value, true_value / fake_value);
      bool res = false;
      if (f1 == f0 && (g1 + g0).norm() < dxNorm)
        res = true;
      if (true_value / fake_value < passrate)
        res = true;
      fmt::print("\tgradient result: {} (dxNorm: {})\n", res, dxNorm);
      return res;
    };
    auto checkJacobian = [&](auto &&f, auto &&g, auto v_c, T ep = 1e-4) {
      constexpr auto passratio = (T)1e-3;

      constexpr int dim = RM_CVREF_T(v_c)::value;
      using VecT = zs::vec<T, dim>;
      VecT x{}, dx{};

      std::random_device rd{};
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(1, 100);
      // check gradient
      for (auto &&[v, dv] : zip(x._data, dx._data)) {
        v = distrib(gen) / (T)100;
        dv = distrib(gen) / (T)100 * ep;
      }

      double eps = dx.abs().max();
      auto x0 = x - dx;
      auto x1 = x + dx;
      auto f0 = f(x0.data());
      auto f1 = f(x1.data());
      auto g0 = g(x0.data());
      auto g1 = g(x1.data());
      double true_value = (f1 - f0 - (g1 + g0) * dx).abs().max() / eps;
      double fake_value = (f1 - f0 - 2 * (g1 + g0) * dx).abs().max() / eps;
      fmt::print(fg(fmt::color::green),
                 "[check jacobian]: realValue: {}, fakeValue: {}, ratio: {}\n",
                 true_value, fake_value, true_value / fake_value);
      bool res = false;
      if (f1 == f0 && (g1 + g0).norm() < eps)
        return true;
      if (true_value / fake_value < passratio)
        res = true;
      fmt::print("\tjacobian result: {} (dxNorm: {})\n", res, eps);
      return res;
    };

    ///
    puts("\tbarrier\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f, g, wrapv<1>{});
      checkJacobian(g, h, wrapv<1>{});
    }
    getchar();
    puts("\tpp\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_pp, g_pp, wrapv<6>{}, 1e-4);
      checkJacobian(g_pp, h_pp, wrapv<6>{}, 1e-4);
    }
    getchar();
    puts("\tpe\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_pe, g_pe, wrapv<9>{}, 1e-4);
      checkJacobian(g_pe, h_pe, wrapv<9>{}, 1e-4);
    }
    getchar();
    puts("\tpt\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_pt, g_pt, wrapv<12>{}, 1e-4);
      checkJacobian(g_pt, h_pt, wrapv<12>{}, 1e-4);
    }
    getchar();
    puts("\tee\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_ee, g_ee, wrapv<12>{}, 1e-4);
      checkJacobian(g_ee, h_ee, wrapv<12>{}, 1e-4);
    }
    getchar();
    puts("\tee mollifier\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_eem, g_eem, wrapv<12>{}, 1e-6);
      checkJacobian(g_eem, h_eem, wrapv<12>{}, 1e-6);
    }
    getchar();
    puts("\telasticity\n");
    for (int i = 0; i < 5; ++i) {
      checkGrad(f_e, g_e, wrapv<12>{}, 1e-4);
      // checkJacobian(g_ee, h_ee, wrapv<12>{}, 1e-4);
    }
    getchar();

    /// construct gradient, prepare hessian
    bool resGrad = A.check_gradient(cudaPol, 1e-4, 1e-3);
    bool resJac = A.check_jacobian(cudaPol, 1e-4, 1e-3);
    fmt::print("check gradient: {}\n", resGrad ? "pass" : "fail");
    fmt::print("check jacobian: {}\n", resJac ? "pass" : "fail");
    getchar();

#if 0
    // rotate gradient and project
    cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                      verts = proxy<space>({}, verts),
                                      dt] __device__(int i) mutable {
      auto grad =
          verts.pack<3, 3>("BCbasis", i).transpose() * vtemp.pack<3>("grad", i);
      if (int BCorder = (verts("BCorder", i)); BCorder > 0)
        for (int d = 0; d != BCorder; ++d)
          grad(d) = 0;
      vtemp.tuple<3>("grad", i) = grad;
    });
#endif
    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ChkIpcSystem, {{"ZSParticles", {"float", "dt", "0.01"}},
                          {"ZSParticles"},
                          {{"int", "grav_inertia", "0"},
                           {"int", "elasticity", "1"},
                           {"int", "barrier", "0"},
                           {"int", "ground_barrier", "0"}},
                          {"FEM"}});

} // namespace zeno