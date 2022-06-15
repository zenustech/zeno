#include "../Structures.hpp"
// #include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/execution/ExecutionPolicy.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/PoissonDisk.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/geometry/VdbLevelSet.h"
#include "zensim/geometry/VdbSampler.h"
#include "zensim/io/MeshIO.hpp"
#include "zensim/math/bit/Bits.h"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

template <typename TileVecT, int codim = 3>
auto retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol,
                               const TileVecT &vtemp,
                               const zs::SmallString &xTag,
                               const typename ZenoParticles::particles_t &eles,
                               zs::wrapv<codim>)
    -> zs::Vector<zs::AABBBox<3, typename TileVecT::value_type>> {
  using namespace zs;
  using T = typename TileVecT::value_type;
  using bv_t = AABBBox<3, T>;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = execspace_e::cuda;
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{},
                               xTag] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = vtemp.template pack<3>(xTag, inds[0]);
    bv_t bv{x0, x0};
    for (int d = 1; d != dim; ++d)
      merge(bv, vtemp.template pack<3>(xTag, inds[d]));
    bvs[ei] = bv;
  });
  return ret;
}
template <typename TileVecT0, typename TileVecT1, int codim = 3>
auto retrieve_bounding_volumes(zs::CudaExecutionPolicy &pol,
                               const TileVecT0 &verts,
                               const zs::SmallString &xTag,
                               const typename ZenoParticles::particles_t &eles,
                               zs::wrapv<codim>, const TileVecT1 &vtemp,
                               const zs::SmallString &dirTag, float stepSize)
    -> zs::Vector<zs::AABBBox<3, typename TileVecT0::value_type>> {
  using namespace zs;
  using T = typename TileVecT0::value_type;
  using bv_t = AABBBox<3, T>;
  static_assert(codim >= 1 && codim <= 4, "invalid co-dimension!\n");
  constexpr auto space = execspace_e::cuda;
  Vector<bv_t> ret{eles.get_allocator(), eles.size()};
  pol(zs::range(eles.size()), [eles = proxy<space>({}, eles),
                               bvs = proxy<space>(ret),
                               verts = proxy<space>({}, verts),
                               vtemp = proxy<space>({}, vtemp),
                               codim_v = wrapv<codim>{}, xTag, dirTag,
                               stepSize] ZS_LAMBDA(int ei) mutable {
    constexpr int dim = RM_CVREF_T(codim_v)::value;
    auto inds =
        eles.template pack<dim>("inds", ei).template reinterpret_bits<int>();
    auto x0 = verts.template pack<3>(xTag, inds[0]);
    auto dir0 = vtemp.template pack<3>(dirTag, inds[0]);
    bv_t bv{get_bounding_box(x0, x0 + stepSize * dir0)};
    for (int d = 1; d != dim; ++d) {
      auto x = verts.template pack<3>(xTag, inds[d]);
      auto dir = vtemp.template pack<3>(dirTag, inds[d]);
      merge(bv, x);
      merge(bv, x + stepSize * dir);
    }
    bvs[ei] = bv;
  });
  return ret;
}

template <typename VecT>
constexpr bool pt_accd(VecT p, VecT t0, VecT t1, VecT t2, VecT dp, VecT dt0,
                       VecT dt1, VecT dt2, typename VecT::value_type eta,
                       typename VecT::value_type thickness,
                       typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  auto mov = (dt0 + dt1 + dt2 + dp) / 4;
  dt0 -= mov;
  dt1 -= mov;
  dt2 -= mov;
  dp -= mov;
  T dispMag2Vec[3] = {dt0.l2NormSqr(), dt1.l2NormSqr(), dt2.l2NormSqr()};
  T tmp = zs::limits<T>::lowest();
  for (int i = 0; i != 3; ++i)
    if (dispMag2Vec[i] > tmp)
      tmp = dispMag2Vec[i];
  T maxDispMag = dp.norm() + zs::sqrt(tmp);
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
  T toc_prev = toc;
  toc = 0;
  while (true) {
    T tocLowerBound = (1 - eta) * (dist2_cur - thickness * thickness) /
                      ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn pt!\n");

    p += tocLowerBound * dp;
    t0 += tocLowerBound * dt0;
    t1 += tocLowerBound * dt1;
    t2 += tocLowerBound * dt2;
    dist2_cur = dist2_pt_unclassified(p, t0, t1, t2);
    dist_cur = zs::sqrt(dist2_cur);
    if (toc &&
        ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}
template <typename VecT>
constexpr bool
ee_accd(VecT ea0, VecT ea1, VecT eb0, VecT eb1, VecT dea0, VecT dea1, VecT deb0,
        VecT deb1, typename VecT::value_type eta,
        typename VecT::value_type thickness, typename VecT::value_type &toc) {
  using T = typename VecT::value_type;
  auto mov = (dea0 + dea1 + deb0 + deb1) / 4;
  dea0 -= mov;
  dea1 -= mov;
  deb0 -= mov;
  deb1 -= mov;
  T maxDispMag = zs::sqrt(zs::max(dea0.l2NormSqr(), dea1.l2NormSqr())) +
                 zs::sqrt(zs::max(deb0.l2NormSqr(), deb1.l2NormSqr()));
  if (maxDispMag == 0)
    return false;

  T dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
  T dFunc = dist2_cur - thickness * thickness;
  if (dFunc <= 0) {
    // since we ensured other place that all dist smaller than dHat are
    // positive, this must be some far away nearly parallel edges
    T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                 (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
    {
      dist2_cur = zs::limits<T>::max();
      for (const auto &dist : dists)
        if (dist < dist2_cur)
          dist2_cur = dist;
      // dist2_cur = *std::min_element(dists.begin(), dists.end());
    }
    dFunc = dist2_cur - thickness * thickness;
  }
  T dist_cur = zs::sqrt(dist2_cur);
  T gap = eta * dFunc / (dist_cur + thickness);
  T toc_prev = toc;
  toc = 0;
  while (true) {
    T tocLowerBound = (1 - eta) * dFunc / ((dist_cur + thickness) * maxDispMag);
    if (tocLowerBound < 0)
      printf("damn ee!\n");

    ea0 += tocLowerBound * dea0;
    ea1 += tocLowerBound * dea1;
    eb0 += tocLowerBound * deb0;
    eb1 += tocLowerBound * deb1;
    dist2_cur = dist2_ee_unclassified(ea0, ea1, eb0, eb1);
    dFunc = dist2_cur - thickness * thickness;
    if (dFunc <= 0) {
      // since we ensured other place that all dist smaller than dHat are
      // positive, this must be some far away nearly parallel edges
      T dists[] = {(ea0 - eb0).l2NormSqr(), (ea0 - eb1).l2NormSqr(),
                   (ea1 - eb0).l2NormSqr(), (ea1 - eb1).l2NormSqr()};
      {
        dist2_cur = zs::limits<T>::max();
        for (const auto &dist : dists)
          if (dist < dist2_cur)
            dist2_cur = dist;
      }
      dFunc = dist2_cur - thickness * thickness;
    }
    dist_cur = zs::sqrt(dist2_cur);
    if (toc && (dFunc / (dist_cur + thickness) < gap))
      break;

    toc += tocLowerBound;
    if (toc > toc_prev) {
      toc = toc_prev;
      return false;
    }
  }
  return true;
}

struct CodimStepping : INode {
  using T = double;
  using Ti = zs::conditional_t<zs::is_same_v<T, double>, zs::i64, zs::i32>;
  using dtiles_t = zs::TileVector<T, 32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;
  using mat2 = zs::vec<T, 2, 2>;
  using mat3 = zs::vec<T, 3, 3>;
  using pair_t = zs::vec<int, 2>;
  using pair3_t = zs::vec<int, 3>;
  using bvh_t = zs::LBvh<3, 32, int, T>;
  using bv_t = zs::AABBBox<3, T>;

  static constexpr vec3 s_groundNormal{0, 1, 0};

  inline static bool projectDBC = true;
  inline static bool BCsatisfied = false;
  inline static T updateZoneTol = 1e-1;
  inline static T consTol = 1e-2;

  inline static T kappaMax = 1e8;
  inline static T kappaMin = 1e4;
  static constexpr T kappa0 = 1e5;
  inline static T kappa = kappa0;
  inline static T xi = 0; // 1e-2; // 2e-3;
  inline static T dHat = 0.001;
  inline static vec3 extForce;

  /// ref: codim-ipc
  static void
  find_ground_intersection_free_stepsize(zs::CudaExecutionPolicy &pol,
                                         const ZenoParticles &zstets,
                                         const dtiles_t &vtemp, T &stepSize) {
    using namespace zs;
    constexpr T slackness = 0.8;
    constexpr auto space = execspace_e::cuda;

    const auto &verts = zstets.getParticles<true>();

    ///
    // query pt
    zs::Vector<T> finalAlpha{verts.get_allocator(), 1};
    finalAlpha.setVal(stepSize);
    pol(Collapse{verts.size()},
        [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
         // boundary
         gn = s_groundNormal, finalAlpha = proxy<space>(finalAlpha),
         stepSize] ZS_LAMBDA(int vi) mutable {
          // this vert affected by sticky boundary conditions
          if (vtemp("BCorder", vi) == 3)
            return;
          auto dir = vtemp.pack<3>("dir", vi);
          auto coef = gn.dot(dir);
          if (coef < 0) { // impacting direction
            auto x = vtemp.pack<3>("xn", vi);
            auto dist = gn.dot(x);
            auto maxAlpha = (dist * 0.8) / (-coef);
            if (maxAlpha < stepSize)
              atomic_min(exec_cuda, &finalAlpha[0], maxAlpha);
          }
        });
    stepSize = finalAlpha.getVal();
    fmt::print(fg(fmt::color::dark_cyan), "ground alpha: {}\n", stepSize);
  }

  struct FEMSystem {
    ///
    auto getCnts() const {
      return zs::make_tuple(nPP.getVal(), nPE.getVal(), nPT.getVal(),
                            nEE.getVal(), ncsPT.getVal(), ncsEE.getVal());
    }
    void computeConstraints(zs::CudaExecutionPolicy &pol,
                            const zs::SmallString &tag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(Collapse{vtemp.size()},
          [vtemp = proxy<space>({}, vtemp), tag] __device__(int vi) mutable {
            auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
            auto BCtarget = vtemp.pack<3>("BCtarget", vi);
            int BCorder = vtemp("BCorder", vi);
            auto x = BCbasis.transpose() * vtemp.pack<3>(tag, vi);
            int d = 0;
            for (; d != BCorder; ++d)
              vtemp("cons", d, vi) = x[d] - BCtarget[d];
            for (; d != 3; ++d)
              vtemp("cons", d, vi) = 0;
          });
    }
    bool areConstraintsSatisfied(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      computeConstraints(pol, "xn");
      auto res = infNorm(pol, vtemp, "cons");
      return res < 1e-2;
    }
    T constraintResidual(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      if (projectDBC)
        return 0;
      Vector<T> num{vtemp.get_allocator(), numDofs},
          den{vtemp.get_allocator(), numDofs};
      constexpr auto space = execspace_e::cuda;
      pol(Collapse{numDofs},
          [vtemp = proxy<space>({}, vtemp), den = proxy<space>(den),
           num = proxy<space>(num)] __device__(int vi) mutable {
            auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
            auto BCtarget = vtemp.pack<3>("BCtarget", vi);
            int BCorder = vtemp("BCorder", vi);
            auto cons = vtemp.pack<3>("cons", vi);
            auto xt = vtemp.pack<3>("xt", vi);
            T n = 0, d = 0;
            // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq5
            for (int d = 0; d != BCorder; ++d) {
              n += zs::sqr(cons[d]);
              d += zs::sqr(col(BCbasis, d).dot(xt) - BCtarget[d]);
            }
            num[vi] = n;
            den[vi] = d;
          });
      auto nsqr = reduce(pol, num);
      auto dsqr = reduce(pol, den);
      T ret = 0;
      if (dsqr == 0)
        ret = std::sqrt(nsqr);
      else
        ret = std::sqrt(nsqr / dsqr);
      return ret < 1e-6 ? 0 : ret;
    }
    void findCollisionConstraints(zs::CudaExecutionPolicy &pol, T dHat,
                                  T xi = 0) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      const auto dHat2 = dHat * dHat;

      nPP.setVal(0);
      nPE.setVal(0);
      nPT.setVal(0);
      nEE.setVal(0);

      ncsPT.setVal(0);
      ncsEE.setVal(0);

      const auto &tris = eles;
      auto triBvs =
          retrieve_bounding_volumes(pol, vtemp, "xn", tris, wrapv<3>{});
      bvh_t stBvh;
      stBvh.build(pol, triBvs);
      auto edgeBvs =
          retrieve_bounding_volumes(pol, vtemp, "xn", edges, wrapv<2>{});
      bvh_t seBvh;
      seBvh.build(pol, edgeBvs);

      /// pt
      pol(Collapse{verts.size()},
          [eles = proxy<space>({}, eles), verts = proxy<space>({}, verts),
           vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stBvh),
           PP = proxy<space>(PP), nPP = proxy<space>(nPP),
           PE = proxy<space>(PE), nPE = proxy<space>(nPE),
           PT = proxy<space>(PT), nPT = proxy<space>(nPT),
           csPT = proxy<space>(csPT), ncsPT = proxy<space>(ncsPT), dHat, xi,
           thickness = xi + dHat] __device__(int vi) mutable {
            const auto dHat2 = zs::sqr(dHat + xi);
            auto p = vtemp.template pack<3>("xn", vi);
            auto bv = bv_t{get_bounding_box(p - thickness, p + thickness)};
            bvh.iter_neighbors(bv, [&](int stI) {
              auto tri = eles.template pack<3>("inds", stI)
                             .template reinterpret_bits<int>();
              if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                return;
              // all affected by sticky boundary conditions
              if (vtemp("BCorder", vi) == 3 && vtemp("BCorder", tri[0]) == 3 &&
                  vtemp("BCorder", tri[1]) == 3 &&
                  vtemp("BCorder", tri[2]) == 3)
                return;
              // ccd
              auto t0 = vtemp.template pack<3>("xn", tri[0]);
              auto t1 = vtemp.template pack<3>("xn", tri[1]);
              auto t2 = vtemp.template pack<3>("xn", tri[2]);

              switch (pt_distance_type(p, t0, t1, t2)) {
              case 0: {
                if (dist2_pp(p, t0) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{vi, tri[0]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 1: {
                if (dist2_pp(p, t1) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{vi, tri[1]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 2: {
                if (dist2_pp(p, t2) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPP[0], 1);
                  PP[no] = pair_t{vi, tri[2]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 3: {
                if (dist2_pe(p, t0, t1) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{vi, tri[0], tri[1]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 4: {
                if (dist2_pe(p, t1, t2) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{vi, tri[1], tri[2]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 5: {
                if (dist2_pe(p, t2, t0) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPE[0], 1);
                  PE[no] = pair3_t{vi, tri[2], tri[0]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              case 6: {
                if (dist2_pt(p, t0, t1, t2) < dHat2) {
                  auto no = atomic_add(exec_cuda, &nPT[0], 1);
                  PT[no] = pair4_t{vi, tri[0], tri[1], tri[2]};
                  csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
                }
                break;
              }
              default:
                break;
              }
            });
          });
      /// ee
      pol(Collapse{edges.size()}, [edges = proxy<space>({}, edges),
                                   verts = proxy<space>({}, verts),
                                   vtemp = proxy<space>({}, vtemp),
                                   bvh = proxy<space>(seBvh),
                                   PP = proxy<space>(PP),
                                   nPP = proxy<space>(nPP),
                                   PE = proxy<space>(PE),
                                   nPE = proxy<space>(nPE),
                                   EE = proxy<space>(EE),
                                   nEE = proxy<space>(nEE),
                                   csEE = proxy<space>(csEE),
                                   ncsEE = proxy<space>(ncsEE), dHat, xi,
                                   thickness =
                                       xi + dHat] __device__(int sei) mutable {
        const auto dHat2 = zs::sqr(dHat + xi);
        auto eiInds = edges.template pack<2>("inds", sei)
                          .template reinterpret_bits<int>();
        bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 &&
                         vtemp("BCorder", eiInds[1]) == 3;
        auto v0 = vtemp.template pack<3>("xn", eiInds[0]);
        auto v1 = vtemp.template pack<3>("xn", eiInds[1]);
        auto rv0 = verts.template pack<3>("x0", eiInds[0]);
        auto rv1 = verts.template pack<3>("x0", eiInds[1]);
        auto [mi, ma] = get_bounding_box(v0, v1);
        auto bv = bv_t{mi - thickness, ma + thickness};
        bvh.iter_neighbors(bv, [&](int sej) {
          if (sei < sej)
            return;
          auto ejInds = edges.template pack<2>("inds", sej)
                            .template reinterpret_bits<int>();
          if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
              eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
            return;
          // all affected by sticky boundary conditions
          if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 &&
              vtemp("BCorder", ejInds[1]) == 3)
            return;
          // ccd
          auto v2 = vtemp.template pack<3>("xn", ejInds[0]);
          auto v3 = vtemp.template pack<3>("xn", ejInds[1]);
          auto rv2 = verts.template pack<3>("x0", ejInds[0]);
          auto rv3 = verts.template pack<3>("x0", ejInds[1]);

          switch (ee_distance_type(v0, v1, v2, v3)) {
          case 0: {
            if (dist2_pp(v0, v2) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[0], ejInds[0]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 1: {
            if (dist2_pp(v0, v3) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[0], ejInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 2: {
            if (dist2_pe(v0, v2, v3) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{eiInds[0], ejInds[0], ejInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 3: {
            if (dist2_pp(v1, v2) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[1], ejInds[0]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 4: {
            if (dist2_pp(v1, v3) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[1], ejInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 5: {
            if (dist2_pe(v1, v2, v3) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{eiInds[1], ejInds[0], ejInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 6: {
            if (dist2_pe(v2, v0, v1) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{ejInds[0], eiInds[0], eiInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 7: {
            if (dist2_pe(v3, v0, v1) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{ejInds[1], eiInds[0], eiInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          case 8: {
            if (dist2_ee(v0, v1, v2, v3) < dHat2) {
              {
                auto no = atomic_add(exec_cuda, &nEE[0], 1);
                EE[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
              }
            }
            break;
          }
          default:
            break;
          }
        });
      });
    }
    void findCCDConstraints(zs::CudaExecutionPolicy &pol, T alpha, T xi = 0) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      const auto dHat2 = dHat * dHat;

      ncsPT.setVal(0);
      ncsEE.setVal(0);

      const auto &tris = eles;
      auto triBvs = retrieve_bounding_volumes(pol, vtemp, "xn", tris,
                                              wrapv<3>{}, vtemp, "dir", alpha);
      bvh_t stBvh;
      stBvh.build(pol, triBvs);
      auto edgeBvs = retrieve_bounding_volumes(pol, vtemp, "xn", edges,
                                               wrapv<2>{}, vtemp, "dir", alpha);
      bvh_t seBvh;
      seBvh.build(pol, edgeBvs);

      /// pt
      pol(Collapse{verts.size()},
          [eles = proxy<space>({}, eles), verts = proxy<space>({}, verts),
           vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(stBvh),
           PP = proxy<space>(PP), nPP = proxy<space>(nPP),
           PE = proxy<space>(PE), nPE = proxy<space>(nPE),
           PT = proxy<space>(PT), nPT = proxy<space>(nPT),
           csPT = proxy<space>(csPT), ncsPT = proxy<space>(ncsPT), xi,
           alpha] __device__(int vi) mutable {
            auto p = vtemp.template pack<3>("xn", vi);
            auto dir = vtemp.template pack<3>("dir", vi);
            auto bv = bv_t{get_bounding_box(p, p + alpha * dir)};
            bv._min -= xi;
            bv._max += xi;
            bvh.iter_neighbors(bv, [&](int stI) {
              auto tri = eles.template pack<3>("inds", stI)
                             .template reinterpret_bits<int>();
              if (vi == tri[0] || vi == tri[1] || vi == tri[2])
                return;
              // all affected by sticky boundary conditions
              if (vtemp("BCorder", vi) == 3 && vtemp("BCorder", tri[0]) == 3 &&
                  vtemp("BCorder", tri[1]) == 3 &&
                  vtemp("BCorder", tri[2]) == 3)
                return;
              csPT[atomic_add(exec_cuda, &ncsPT[0], 1)] = pair_t{vi, stI};
            });
          });
      /// ee
      pol(Collapse{edges.size()},
          [edges = proxy<space>({}, edges), verts = proxy<space>({}, verts),
           vtemp = proxy<space>({}, vtemp), bvh = proxy<space>(seBvh),
           PP = proxy<space>(PP), nPP = proxy<space>(nPP),
           PE = proxy<space>(PE), nPE = proxy<space>(nPE),
           EE = proxy<space>(PT), nEE = proxy<space>(nPT),
           csEE = proxy<space>(csEE), ncsEE = proxy<space>(ncsEE), xi,
           alpha] __device__(int sei) mutable {
            auto eiInds = edges.template pack<2>("inds", sei)
                              .template reinterpret_bits<int>();
            bool selfFixed = vtemp("BCorder", eiInds[0]) == 3 &&
                             vtemp("BCorder", eiInds[1]) == 3;
            auto v0 = vtemp.template pack<3>("xn", eiInds[0]);
            auto v1 = vtemp.template pack<3>("xn", eiInds[1]);
            auto dir0 = vtemp.template pack<3>("dir", eiInds[0]);
            auto dir1 = vtemp.template pack<3>("dir", eiInds[1]);
            auto bv = bv_t{get_bounding_box(v0, v0 + alpha * dir0)};
            merge(bv, v1);
            merge(bv, v1 + alpha * dir1);
            bv._min -= xi;
            bv._max += xi;
            bvh.iter_neighbors(bv, [&](int sej) {
              if (sei < sej)
                return;
              auto ejInds = edges.template pack<2>("inds", sej)
                                .template reinterpret_bits<int>();
              if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
                  eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
                return;
              // all affected by sticky boundary conditions
              if (selfFixed && vtemp("BCorder", ejInds[0]) == 3 &&
                  vtemp("BCorder", ejInds[1]) == 3)
                return;
              csEE[atomic_add(exec_cuda, &ncsEE[0], 1)] = pair_t{sei, sej};
            });
          });
    }
    ///
    void computeBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol,
                                          const zs::SmallString &gTag = "grad");

    void intersectionFreeStepsize(zs::CudaExecutionPolicy &pol, T xi,
                                  T &stepSize) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;

      Vector<T> alpha{verts.get_allocator(), 1};
      alpha.setVal(stepSize);
      auto npt = ncsPT.getVal();
      pol(range(npt),
          [csPT = proxy<space>(csPT), vtemp = proxy<space>({}, vtemp),
           eles = proxy<space>({}, eles), alpha = proxy<space>(alpha), stepSize,
           xi] __device__(int pti) {
            auto ids = csPT[pti];
            int vi = ids[0];
            auto tri = eles.template pack<3>("inds", ids[1])
                           .template reinterpret_bits<int>();
            auto p = vtemp.template pack<3>("xn", vi);
            auto t0 = vtemp.template pack<3>("xn", tri[0]);
            auto t1 = vtemp.template pack<3>("xn", tri[1]);
            auto t2 = vtemp.template pack<3>("xn", tri[2]);
            auto dp = vtemp.template pack<3>("dir", vi);
            auto dt0 = vtemp.template pack<3>("dir", tri[0]);
            auto dt1 = vtemp.template pack<3>("dir", tri[1]);
            auto dt2 = vtemp.template pack<3>("dir", tri[2]);
            T tmp = stepSize;
            if (pt_accd(p, t0, t1, t2, dp, dt0, dt1, dt2, (T)0.1, xi, tmp))
              atomic_min(exec_cuda, &alpha[0], tmp);
          });
      auto nee = ncsEE.getVal();
      pol(range(nee),
          [csEE = proxy<space>(csEE), vtemp = proxy<space>({}, vtemp),
           edges = proxy<space>({}, edges), alpha = proxy<space>(alpha),
           stepSize, xi] __device__(int eei) {
            auto ids = csEE[eei];
            auto ei = edges.template pack<2>("inds", ids[0])
                          .template reinterpret_bits<int>();
            auto ej = edges.template pack<2>("inds", ids[1])
                          .template reinterpret_bits<int>();
            auto ea0 = vtemp.template pack<3>("xn", ei[0]);
            auto ea1 = vtemp.template pack<3>("xn", ei[1]);
            auto eb0 = vtemp.template pack<3>("xn", ej[0]);
            auto eb1 = vtemp.template pack<3>("xn", ej[1]);
            auto dea0 = vtemp.template pack<3>("dir", ei[0]);
            auto dea1 = vtemp.template pack<3>("dir", ei[1]);
            auto deb0 = vtemp.template pack<3>("dir", ej[0]);
            auto deb1 = vtemp.template pack<3>("dir", ej[1]);
            auto tmp = stepSize;
            if (ee_accd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, (T)0.1, xi,
                        tmp))
              atomic_min(exec_cuda, &alpha[0], tmp);
          });
      stepSize = alpha.getVal();
    }
    ///
    void computeBoundaryBarrierGradientAndHessian(
        zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag = "grad") {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(range(vtemp.size()),
          [verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
           tempPB = proxy<space>({}, tempPB), gTag, gn = s_groundNormal,
           dHat2 = dHat * dHat, kappa = kappa,
           projectDBC = projectDBC] ZS_LAMBDA(int vi) mutable {
            auto x = vtemp.pack<3>("xn", vi);
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

            // make_pd(hess);
            mat3 BCbasis[1] = {vtemp.pack<3, 3>("BCbasis", vi)};
            int BCorder[1] = {(int)vtemp("BCorder", vi)};
            int BCfixed[1] = {(int)vtemp("BCfixed", vi)};
            rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
            tempPB.tuple<9>("H", vi) = hess;
            for (int i = 0; i != 3; ++i)
              for (int j = 0; j != 3; ++j) {
                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, vi), hess(i, j));
              }
          });
      return;
    }
    template <typename Model>
    void
    computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol,
                                     const Model &model,
                                     const zs::SmallString &gTag = "grad") {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      cudaPol(zs::range(eles.size()),
              [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, etemp),
               verts = proxy<space>({}, verts), eles = proxy<space>({}, eles),
               model, gTag, dt = this->dt,
               projectDBC = projectDBC] __device__(int ei) mutable {
                auto IB = eles.template pack<2, 2>("IB", ei);
                auto inds = eles.template pack<3>("inds", ei)
                                .template reinterpret_bits<int>();
                auto vole = eles("vol", ei);
                vec3 xs[3] = {vtemp.template pack<3>("xn", inds[0]),
                              vtemp.template pack<3>("xn", inds[1]),
                              vtemp.template pack<3>("xn", inds[2])};
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];

                mat3 BCbasis[3];
                int BCorder[3];
                int BCfixed[3];
                for (int i = 0; i != 3; ++i) {
                  BCbasis[i] = vtemp.pack<3, 3>("BCbasis", inds[i]);
                  BCorder[i] = vtemp("BCorder", inds[i]);
                  BCfixed[i] = vtemp("BCfixed", inds[i]);
                }

                zs::vec<T, 9, 9> H;
#if 0
      mat2 A{}, temp{};
      auto dA_div_dx = zs::vec<T, 4, 9>::zeros();
      A(0, 0) = x1x0.l2NormSqr();
      A(1, 0) = A(0, 1) = x1x0.dot(x2x0);
      A(1, 1) = x2x0.l2NormSqr();

      auto IA = inverse(A);
      auto lnJ = zs::log(determinant(A) * determinant(IB)) / 2;
      temp = -dt * dt * vole *
             (model.mu / 2 * IB + (-model.mu + model.lam * lnJ) / 2 * IA);
      for (int d = 0; d != 3; ++d) {
        dA_div_dx(0, d) = -2 * x1x0[d];
        dA_div_dx(0, 3 + d) = 2 * x1x0[d];
        dA_div_dx(1, d) = -x1x0[d] - x2x0[d];
        dA_div_dx(1, 3 + d) = x2x0[d];
        dA_div_dx(1, 6 + d) = x1x0[d];
        dA_div_dx(2, d) = -x1x0[d] - x2x0[d];
        dA_div_dx(2, 3 + d) = x2x0[d];
        dA_div_dx(2, 6 + d) = x1x0[d];
        dA_div_dx(3, d) = -2 * x2x0[d];
        dA_div_dx(3, 6 + d) = 2 * x2x0[d];
      }

      for (int i_ = 0; i_ != 3; ++i_) {
        auto vi = inds[i_];
        for (int d = 0; d != 3; ++d) {
          int i = i_ * 3 + d;
          atomic_add(
              exec_cuda, &vtemp(gTag, d, vi),
              dA_div_dx(0, i) * temp(0, 0) + dA_div_dx(1, i) * temp(1, 0) +
                  dA_div_dx(2, i) * temp(0, 1) + dA_div_dx(3, i) * temp(1, 1));
        }
      }

      // hessian rotation: trans^T hess * trans
      // left trans^T: multiplied on rows
      // right trans: multiplied on cols
      
      using mat9 = zs::vec<T, 9, 9>;
      mat9 ahess[4];
      for (int i = 0; i != 4; ++i)
        ahess[i] = mat9::zeros();
      for (int i = 0; i != 3; ++i) {
        ahess[3](i, i) = ahess[0](i, i) = 2;
        ahess[3](6 + i, 6 + i) = ahess[0](3 + i, 3 + i) = 2;
        ahess[3](i, 6 + i) = ahess[0](i, 3 + i) = -2;
        ahess[3](6 + i, i) = ahess[0](3 + i, i) = -2;
      }
      for (int i = 0; i != 3; ++i) {
        ahess[2](3 + i, 6 + i) = ahess[1](3 + i, 6 + i) = 1;
        ahess[2](6 + i, 3 + i) = ahess[1](6 + i, 3 + i) = 1;
        ahess[2](i, 3 + i) = ahess[1](i, 3 + i) = -1;
        ahess[2](i, 6 + i) = ahess[1](i, 6 + i) = -1;
        ahess[2](3 + i, i) = ahess[1](3 + i, i) = -1;
        ahess[2](6 + i, i) = ahess[1](6 + i, i) = -1;
        ahess[2](i, i) = ahess[1](i, i) = 2;
      }

      zs::vec<T, 9> ainvda;
      for (int i_ = 0; i_ < 3; ++i_) {
        for (int d = 0; d < 3; ++d) {
          int i = i_ * 3 + d;
          ainvda(i) = dA_div_dx(0, i) * IA(0, 0) + dA_div_dx(1, i) * IA(1, 0) +
                      dA_div_dx(2, i) * IA(0, 1) + dA_div_dx(3, i) * IA(1, 1);

          const T deta = determinant(A);
          const T lnJ = zs::log(deta * determinant(IB)) / 2;
          const T term1 = (-model.mu + model.lam * lnJ) / 2;
          H = (-term1 + model.lam / 4) * dyadic_prod(ainvda, ainvda);

          zs::vec<T, 4, 9> aderivadj;
          for (int d = 0; d != 9; ++d) {
            aderivadj(0, d) = dA_div_dx(3, d);
            aderivadj(1, d) = -dA_div_dx(1, d);
            aderivadj(2, d) = -dA_div_dx(2, d);
            aderivadj(3, d) = dA_div_dx(0, d);
          }
          H += term1 / deta * aderivadj.transpose() * dA_div_dx;

          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
              H += (term1 * IA(i, j) + model.mu / 2 * IB(i, j)) *
                   ahess[i + j * 2];
            }
        }
      }
      make_pd(H);
#else
              zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1],
                                  x2x0[1], x1x0[2], x2x0[2]};
              auto F = Ds * IB;

              auto dFdX = dFdXMatrix(IB, wrapv<3>{});
              auto dFdXT = dFdX.transpose();
              auto f0 = col(F, 0);
              auto f1 = col(F, 1);
              auto f0Norm = zs::sqrt(f0.l2NormSqr());
              auto f1Norm = zs::sqrt(f1.l2NormSqr());
              auto f0Tf1 = f0.dot(f1);
              zs::vec<T, 3, 2> Pstretch, Pshear;
              for (int d = 0; d != 3; ++d) {
                Pstretch(d, 0) = 2 * (1 - 1 / f0Norm) * F(d, 0);
                Pstretch(d, 1) = 2 * (1 - 1 / f1Norm) * F(d, 1);
                Pshear(d, 0) = 2 * f0Tf1 * f1(d);
                Pshear(d, 1) = 2 * f0Tf1 * f0(d);
              }
              auto vecP = flatten(Pstretch + Pshear);
              auto vfdt2 = -vole * (dFdXT * vecP) * (model.mu * dt * dt);

              for (int i = 0; i != 3; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &vtemp(gTag, d, vi),
                             (T)vfdt2(i * 3 + d));
              }

              /// ref: A Finite Element Formulation of Baraff-Witkin Cloth
              // suggested by huang kemeng
              auto stretchHessian = [&F, &model]() {
                auto H = zs::vec<T, 6, 6>::zeros();
                const zs::vec<T, 2> u{1, 0};
                const zs::vec<T, 2> v{0, 1};
                const T I5u = (F * u).l2NormSqr();
                const T I5v = (F * v).l2NormSqr();
                const T invSqrtI5u = (T)1 / zs::sqrt(I5u);
                const T invSqrtI5v = (T)1 / zs::sqrt(I5v);

                H(0, 0) = H(1, 1) = H(2, 2) = zs::max(1 - invSqrtI5u, (T)0);
                H(3, 3) = H(4, 4) = H(5, 5) = zs::max(1 - invSqrtI5v, (T)0);

                const auto fu = col(F, 0).normalized();
                const T uCoeff = (1 - invSqrtI5u >= 0) ? invSqrtI5u : (T)1;
                for (int i = 0; i != 3; ++i)
                  for (int j = 0; j != 3; ++j)
                    H(i, j) += uCoeff * fu(i) * fu(j);

                const auto fv = col(F, 1).normalized();
                const T vCoeff = (1 - invSqrtI5v >= 0) ? invSqrtI5v : (T)1;
                for (int i = 0; i != 3; ++i)
                  for (int j = 0; j != 3; ++j)
                    H(3 + i, 3 + j) += vCoeff * fv(i) * fv(j);

                H *= model.mu;
                return H;
              };
              auto shearHessian = [&F, &model]() {
                using mat6 = zs::vec<T, 6, 6>;
                auto H = mat6::zeros();
                const zs::vec<T, 2> u{1, 0};
                const zs::vec<T, 2> v{0, 1};
                const T I6 = (F * u).dot(F * v);
                const T signI6 = I6 >= 0 ? 1 : -1;

                H(3, 0) = H(4, 1) = H(5, 2) = H(0, 3) = H(1, 4) = H(2, 5) =
                    (T)1;

                const auto g_ = F * (dyadic_prod(u, v) + dyadic_prod(v, u));
                zs::vec<T, 6> g{};
                for (int j = 0, offset = 0; j != 2; ++j) {
                  for (int i = 0; i != 3; ++i)
                    g(offset++) = g_(i, j);
                }

                const T I2 = F.l2NormSqr();
                const T lambda0 =
                    (T)0.5 * (I2 + zs::sqrt(I2 * I2 + (T)12 * I6 * I6));

                const zs::vec<T, 6> q0 =
                    (I6 * H * g + lambda0 * g).normalized();

                auto t = mat6::identity();
                t = 0.5 * (t + signI6 * H);

                const zs::vec<T, 6> Tq = t * q0;
                const auto normTq = Tq.l2NormSqr();

                mat6 dPdF = zs::abs(I6) * (t - (dyadic_prod(Tq, Tq) / normTq)) +
                            lambda0 * (dyadic_prod(q0, q0));
                dPdF *= model.mu;
                return dPdF;
              };
              auto He = stretchHessian() + shearHessian();
              H = dFdX.transpose() * He * dFdX;
#endif
                H *= dt * dt * vole;

                // rotate and project
                rotate_hessian(H, BCbasis, BCorder, BCfixed, projectDBC);
                etemp.tuple<9 * 9>("He", ei) = H;
                for (int vi = 0; vi != 3; ++vi) {
                  for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j) {
                      atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]),
                                 H(vi * 3 + i, vi * 3 + j));
                    }
                }
              });
    }
    template <typename Pol, typename Model>
    T energy(Pol &pol, const Model &model, const zs::SmallString tag,
             bool includeAugLagEnergy = false) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
      pol(range(vtemp.size()), [verts = proxy<space>({}, verts),
                                vtemp = proxy<space>({}, vtemp),
                                res = proxy<space>(res), tag,
                                extForce = extForce,
                                dt = this->dt] __device__(int vi) mutable {
        // inertia
        auto m = verts("m", vi);
        auto x = vtemp.pack<3>(tag, vi);
        atomic_add(exec_cuda, &res[0],
                   (T)0.5 * m * (x - vtemp.pack<3>("xtilde", vi)).l2NormSqr());
        // gravity
        atomic_add(exec_cuda, &res[0],
                   -m * extForce.dot(x - vtemp.pack<3>("xt", vi)) * dt * dt);
      });
      // elasticity
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model,
                               dt = this->dt] __device__(int ei) mutable {
        auto IB = eles.template pack<2, 2>("IB", ei);
        auto inds =
            eles.template pack<3>("inds", ei).template reinterpret_bits<int>();
        auto vole = eles("vol", ei);
        vec3 xs[3] = {vtemp.template pack<3>(tag, inds[0]),
                      vtemp.template pack<3>(tag, inds[1]),
                      vtemp.template pack<3>(tag, inds[2])};
        mat2 A{};
        T E;
        auto x1x0 = xs[1] - xs[0];
        auto x2x0 = xs[2] - xs[0];
#if 0
        {
          A(0, 0) = x1x0.l2NormSqr();
          A(1, 0) = A(0, 1) = x1x0.dot(x2x0);
          A(1, 1) = x2x0.l2NormSqr();

          auto IA = inverse(A);
          auto lnJ = zs::log(determinant(A) * determinant(IB)) / 2;
          E = dt * dt * vole *
              (model.mu / 2 * (trace(IB * A) - 2 - 2 * lnJ) +
               model.lam / 2 * lnJ * lnJ);
        }
#else
        zs::vec<T, 3, 2> Ds{x1x0[0], x2x0[0], x1x0[1], x2x0[1], x1x0[2], x2x0[2]};
        auto F = Ds * IB;
        auto f0 = col(F, 0);
        auto f1 = col(F, 1);
        auto f0Norm = zs::sqrt(f0.l2NormSqr());
        auto f1Norm = zs::sqrt(f1.l2NormSqr());
        auto Estretch = dt * dt * model.mu *vole * (zs::sqr(f0Norm - 1) + zs::sqr(f1Norm - 1));
        auto Eshear = dt * dt * model.mu *vole * zs::sqr(f0.dot(f1));
        E = Estretch + Eshear;
#endif
        atomic_add(exec_cuda, &res[0], E);
      });
      // contacts
      {
        auto activeGap2 = dHat * dHat + 2 * xi * dHat;
        auto numPP = nPP.getVal();
        pol(range(numPP),
            [vtemp = proxy<space>({}, vtemp), tempPP = proxy<space>({}, tempPP),
             PP = proxy<space>(PP), res = proxy<space>(res), xi2 = xi * xi,
             dHat = dHat, activeGap2,
             kappa = kappa] __device__(int ppi) mutable {
              auto pp = PP[ppi];
              auto x0 = vtemp.pack<3>("xn", pp[0]);
              auto x1 = vtemp.pack<3>("xn", pp[1]);
              auto dist2 = dist2_pp(x0, x1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              atomic_add(exec_cuda, &res[0],
                         zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numPE = nPE.getVal();
        pol(range(numPE),
            [vtemp = proxy<space>({}, vtemp), tempPE = proxy<space>({}, tempPE),
             PE = proxy<space>(PE), res = proxy<space>(res), xi2 = xi * xi,
             dHat = dHat, activeGap2,
             kappa = kappa] __device__(int pei) mutable {
              auto pe = PE[pei];
              auto p = vtemp.pack<3>("xn", pe[0]);
              auto e0 = vtemp.pack<3>("xn", pe[1]);
              auto e1 = vtemp.pack<3>("xn", pe[2]);

              auto dist2 = dist2_pe(p, e0, e1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              atomic_add(exec_cuda, &res[0],
                         zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numPT = nPT.getVal();
        pol(range(numPT),
            [vtemp = proxy<space>({}, vtemp), tempPT = proxy<space>({}, tempPT),
             PT = proxy<space>(PT), res = proxy<space>(res), xi2 = xi * xi,
             dHat = dHat, activeGap2,
             kappa = kappa] __device__(int pti) mutable {
              auto pt = PT[pti];
              auto p = vtemp.pack<3>("xn", pt[0]);
              auto t0 = vtemp.pack<3>("xn", pt[1]);
              auto t1 = vtemp.pack<3>("xn", pt[2]);
              auto t2 = vtemp.pack<3>("xn", pt[3]);

              auto dist2 = dist2_pt(p, t0, t1, t2);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              atomic_add(exec_cuda, &res[0],
                         zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        auto numEE = nEE.getVal();
        pol(range(numEE),
            [vtemp = proxy<space>({}, vtemp), tempEE = proxy<space>({}, tempEE),
             EE = proxy<space>(EE), res = proxy<space>(res), xi2 = xi * xi,
             dHat = dHat, activeGap2,
             kappa = kappa] __device__(int eei) mutable {
              auto ee = EE[eei];
              auto ea0 = vtemp.pack<3>("xn", ee[0]);
              auto ea1 = vtemp.pack<3>("xn", ee[1]);
              auto eb0 = vtemp.pack<3>("xn", ee[2]);
              auto eb1 = vtemp.pack<3>("xn", ee[3]);

              auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              atomic_add(exec_cuda, &res[0],
                         zs::barrier(dist2 - xi2, activeGap2, kappa));
            });
        // boundary
        pol(range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), res = proxy<space>(res),
             gn = s_groundNormal, dHat2 = dHat * dHat,
             kappa = kappa] ZS_LAMBDA(int vi) mutable {
              auto x = vtemp.pack<3>("xn", vi);
              auto dist = gn.dot(x);
              auto dist2 = dist * dist;
              if (dist2 < dHat2) {
                auto temp = -(dist2 - dHat2) * (dist2 - dHat2) *
                            zs::log(dist2 / dHat2) * kappa;
                atomic_add(exec_cuda, &res[0], temp);
              }
            });
      }
      // constraints
      if (includeAugLagEnergy) {
        computeConstraints(pol, tag);
        pol(range(numDofs), [vtemp = proxy<space>({}, vtemp),
                             res = proxy<space>(res), tag,
                             kappa = kappa] __device__(int vi) mutable {
          auto x = vtemp.pack<3>(tag, vi);
          auto cons = vtemp.template pack<3>("cons", vi);
          auto w = vtemp("ws", vi);
          auto lambda = vtemp.pack<3>("lambda", vi);
          atomic_add(exec_cuda, &res[0],
                     (T)(-lambda.dot(cons) + 0.5 * kappa * cons.l2NormSqr()));
        });
      }
      return res.getVal();
    }
    template <typename Pol> void project(Pol &pol, const zs::SmallString tag) {
#if 1
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // projection
      pol(zs::range(verts.size()),
          [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
           tag] ZS_LAMBDA(int vi) mutable {
            auto BCbasis = vtemp.template pack<3, 3>("BCbasis", vi);
            int BCorder = vtemp("BCorder", vi);
            for (int d = 0; d != BCorder; ++d)
              vtemp(tag, d, vi) = 0;
            // if (verts("x", 1, vi) > 0.8)
            //  vtemp.tuple<3>(tag, vi) = vec3::zeros();
          });
#endif
    }
    template <typename Pol>
    void precondition(Pol &pol, const zs::SmallString srcTag,
                      const zs::SmallString dstTag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // precondition
      pol(zs::range(verts.size()), [vtemp = proxy<space>({}, vtemp), srcTag,
                                    dstTag] ZS_LAMBDA(int vi) mutable {
        vtemp.template tuple<3>(dstTag, vi) =
            vtemp.template pack<3, 3>("P", vi) *
            vtemp.template pack<3>(srcTag, vi);
      });
    }
    template <typename Pol>
    void multiply(Pol &pol, const zs::SmallString dxTag,
                  const zs::SmallString bTag) {
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
      pol(range(numVerts), [execTag, verts = proxy<space>({}, verts),
                            vtemp = proxy<space>({}, vtemp), dxTag,
                            bTag] ZS_LAMBDA(int vi) mutable {
        auto m = verts("m", vi);
        auto dx = vtemp.template pack<3>(dxTag, vi);
        auto BCbasis = vtemp.template pack<3, 3>("BCbasis", vi);
        int BCorder = vtemp("BCorder", vi);
        // dx = BCbasis.transpose() * m * BCbasis * dx;
        auto M = mat3::identity() * m;
        M = BCbasis.transpose() * M * BCbasis;
        for (int i = 0; i != BCorder; ++i)
          for (int j = 0; j != BCorder; ++j)
            M(i, j) = (i == j ? 1 : 0);
        dx = M * dx;
        for (int d = 0; d != 3; ++d)
          atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
      });
      // elasticity
      pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                           vtemp = proxy<space>({}, vtemp),
                           eles = proxy<space>({}, eles), dxTag,
                           bTag] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = 3;
        auto inds =
            eles.template pack<3>("inds", ei).template reinterpret_bits<int>();
        zs::vec<T, 3 * dim> temp{};
        for (int vi = 0; vi != 3; ++vi)
          for (int d = 0; d != dim; ++d) {
            temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
          }
        auto He = etemp.template pack<dim * 3, dim * 3>("He", ei);

        temp = He * temp;

        for (int vi = 0; vi != 3; ++vi)
          for (int d = 0; d != dim; ++d) {
            atomic_add(execTag, &vtemp(bTag, d, inds[vi]), temp[vi * dim + d]);
          }
      });
      // contacts
      {
        auto numPP = nPP.getVal();
        pol(range(numPP), [execTag, tempPP = proxy<space>({}, tempPP),
                           vtemp = proxy<space>({}, vtemp), dxTag, bTag,
                           PP = proxy<space>(PP)] ZS_LAMBDA(int ppi) mutable {
          constexpr int dim = 3;
          auto pp = PP[ppi];
          zs::vec<T, dim * 2> temp{};
          for (int vi = 0; vi != 2; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pp[vi]);
            }
          auto ppHess = tempPP.template pack<6, 6>("H", ppi);

          temp = ppHess * temp;

          for (int vi = 0; vi != 2; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pp[vi]), temp[vi * dim + d]);
            }
        });
        auto numPE = nPE.getVal();
        pol(range(numPE), [execTag, tempPE = proxy<space>({}, tempPE),
                           vtemp = proxy<space>({}, vtemp), dxTag, bTag,
                           PE = proxy<space>(PE)] ZS_LAMBDA(int pei) mutable {
          constexpr int dim = 3;
          auto pe = PE[pei];
          zs::vec<T, dim * 3> temp{};
          for (int vi = 0; vi != 3; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pe[vi]);
            }
          auto peHess = tempPE.template pack<9, 9>("H", pei);

          temp = peHess * temp;

          for (int vi = 0; vi != 3; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pe[vi]), temp[vi * dim + d]);
            }
        });
        auto numPT = nPT.getVal();
        pol(range(numPT), [execTag, tempPT = proxy<space>({}, tempPT),
                           vtemp = proxy<space>({}, vtemp), dxTag, bTag,
                           PT = proxy<space>(PT)] ZS_LAMBDA(int pti) mutable {
          constexpr int dim = 3;
          auto pt = PT[pti];
          zs::vec<T, dim * 4> temp{};
          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, pt[vi]);
            }
          auto ptHess = tempPT.template pack<12, 12>("H", pti);

          temp = ptHess * temp;

          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, pt[vi]), temp[vi * dim + d]);
            }
        });
        auto numEE = nEE.getVal();
        pol(range(numEE), [execTag, tempEE = proxy<space>({}, tempEE),
                           vtemp = proxy<space>({}, vtemp), dxTag, bTag,
                           EE = proxy<space>(EE)] ZS_LAMBDA(int eei) mutable {
          constexpr int dim = 3;
          auto ee = EE[eei];
          zs::vec<T, dim * 4> temp{};
          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              temp[vi * dim + d] = vtemp(dxTag, d, ee[vi]);
            }
          auto eeHess = tempEE.template pack<12, 12>("H", eei);

          temp = eeHess * temp;

          for (int vi = 0; vi != 4; ++vi)
            for (int d = 0; d != dim; ++d) {
              atomic_add(execTag, &vtemp(bTag, d, ee[vi]), temp[vi * dim + d]);
            }
        });
        // boundary
        pol(range(verts.size()), [execTag, vtemp = proxy<space>({}, vtemp),
                                  tempPB = proxy<space>({}, tempPB), dxTag,
                                  bTag] ZS_LAMBDA(int vi) mutable {
          auto dx = vtemp.template pack<3>(dxTag, vi);
          auto pbHess = tempPB.template pack<3, 3>("H", vi);
          dx = pbHess * dx;
          for (int d = 0; d != 3; ++d)
            atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
        });
      } // end contacts

      // constraint hessian
      if (!BCsatisfied) {
        pol(range(vtemp.size()),
            [execTag, vtemp = proxy<space>({}, vtemp), dxTag, bTag,
             kappa = kappa] ZS_LAMBDA(int vi) mutable {
              auto cons = vtemp.template pack<3>("cons", vi);
              auto dx = vtemp.template pack<3>(dxTag, vi);
              auto w = vtemp("ws", vi);
              for (int d = 0; d != 3; ++d)
                if (cons[d] != 0)
                  atomic_add(execTag, &vtemp(bTag, d, vi), kappa * w * dx(d));
            });
      }
    }

    FEMSystem(const dtiles_t &verts, const tiles_t &edges, const tiles_t &eles,
              const tiles_t &coVerts, const tiles_t &coEles, dtiles_t &vtemp,
              dtiles_t &etemp, T dt, const ZenoConstitutiveModel &models)
        : verts{verts}, edges{edges}, eles{eles}, coVerts{coVerts},
          coEles{coEles}, vtemp{vtemp}, etemp{etemp},
          PP{verts.get_allocator(), 100000}, nPP{verts.get_allocator(), 1},
          tempPP{PP.get_allocator(),
                 {{"H", 36}, {"inds_pre", 2}, {"dist2_pre", 1}},
                 100000},
          PE{verts.get_allocator(), 100000}, nPE{verts.get_allocator(), 1},
          tempPE{PE.get_allocator(),
                 {{"H", 81}, {"inds_pre", 3}, {"dist2_pre", 1}},
                 100000},
          PT{verts.get_allocator(), 100000}, nPT{verts.get_allocator(), 1},
          tempPT{PT.get_allocator(),
                 {{"H", 144}, {"inds_pre", 4}, {"dist2_pre", 1}},
                 100000},
          EE{verts.get_allocator(), 100000}, nEE{verts.get_allocator(), 1},
          tempEE{EE.get_allocator(),
                 {{"H", 144}, {"inds_pre", 4}, {"dist2_pre", 1}},
                 100000},
          csPT{verts.get_allocator(), 100000}, csEE{verts.get_allocator(),
                                                    100000},
          ncsPT{verts.get_allocator(), 1}, ncsEE{verts.get_allocator(), 1},
          tempPB{verts.get_allocator(), {{"H", 9}}, verts.size()}, dt{dt},
          models{models} {
      coOffset = verts.size();
      numDofs = coOffset + coVerts.size();
      nPP.setVal(0);
      nPE.setVal(0);
      nPT.setVal(0);
      nEE.setVal(0);

      ncsPT.setVal(0);
      ncsEE.setVal(0);
    }

    const dtiles_t &verts;
    std::size_t coOffset, numDofs;
    const tiles_t &edges;
    const tiles_t &eles;
    // (scripted) collision objects
    const tiles_t &coVerts, &coEles;
    dtiles_t &vtemp;
    dtiles_t &etemp;
    // self contacts
    using pair_t = zs::vec<int, 2>;
    using pair3_t = zs::vec<int, 3>;
    using pair4_t = zs::vec<int, 4>;
    zs::Vector<pair_t> PP;
    zs::Vector<int> nPP;
    dtiles_t tempPP;
    zs::Vector<pair3_t> PE;
    zs::Vector<int> nPE;
    dtiles_t tempPE;
    zs::Vector<pair4_t> PT;
    zs::Vector<int> nPT;
    dtiles_t tempPT;
    zs::Vector<pair4_t> EE;
    zs::Vector<int> nEE;
    dtiles_t tempEE;

    zs::Vector<pair_t> csPT, csEE;
    zs::Vector<int> ncsPT, ncsEE;

    // boundary contacts
    dtiles_t tempPB;
    // end contacts
    T dt;
    const ZenoConstitutiveModel &models;
  };

  static T reduce(zs::CudaExecutionPolicy &cudaPol, const zs::Vector<T> &res) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> ret{res.get_allocator(), 1};
    auto sid = cudaPol.getStreamid();
    auto procid = cudaPol.getProcid();
    auto &context = Cuda::context(procid);
    auto stream = (cudaStream_t)context.streamSpare(sid);
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, temp_bytes, res.data(), ret.data(),
                              res.size(), std::plus<T>{}, (T)0, stream);
    Vector<std::max_align_t> temp{res.get_allocator(),
                                  temp_bytes / sizeof(std::max_align_t) + 1};
    cub::DeviceReduce::Reduce(temp.data(), temp_bytes, res.data(), ret.data(),
                              res.size(), std::plus<T>{}, (T)0, stream);
    context.syncStreamSpare(sid);
    return (T)ret.getVal();
  }
  static T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
               const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
#if 0
    Vector<T> ret{vertData.get_allocator(), 1};
    ret.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), ret = proxy<space>(ret), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<3>(tag0, pi);
              auto v1 = data.pack<3>(tag1, pi);
              atomic_add(exec_cuda, &ret[0], v0.dot(v1));
            });
    return (T)ret.getVal();
#else
    Vector<double> res{vertData.get_allocator(), vertData.size()};
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<3>(tag0, pi);
              auto v1 = data.pack<3>(tag1, pi);
              res[pi] = v0.dot(v1);
            });
    return reduce(cudaPol, res);
#endif
  }
  static T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
                   const zs::SmallString tag = "dir") {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> ret{vertData.get_allocator(), 1};
    ret.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), ret = proxy<space>(ret),
             tag] __device__(int pi) mutable {
              auto v = data.pack<3>(tag, pi);
              atomic_max(exec_cuda, &ret[0], v.abs().max());
            });
    return ret.getVal();
  }
  template <
      typename VecT, int N = VecT::template range_t<0>::value,
      zs::enable_if_all<N % 3 == 0, N == VecT::template range_t<1>::value> = 0>
  static constexpr void rotate_hessian(zs::VecInterface<VecT> &H,
                                       mat3 BCbasis[N / 3], int BCorder[N / 3],
                                       int BCfixed[], bool projectDBC) {
    // hessian rotation: trans^T hess * trans
    // left trans^T: multiplied on rows
    // right trans: multiplied on cols
    constexpr int NV = N / 3;
    // rotate and project
    for (int vi = 0; vi != NV; ++vi) {
      int offsetI = vi * 3;
      for (int vj = 0; vj != NV; ++vj) {
        int offsetJ = vj * 3;
        mat3 tmp{};
        for (int i = 0; i != 3; ++i)
          for (int j = 0; j != 3; ++j)
            tmp(i, j) = H(offsetI + i, offsetJ + j);
        // rotate
        tmp = BCbasis[vi].transpose() * tmp * BCbasis[vj];
        // project
        if (projectDBC || (!projectDBC && (BCfixed[vi] > 0 || BCfixed[vj] > 0)))
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
    return;
  }

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    std::shared_ptr<ZenoParticles> zsboundary;
    if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
      zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");
    auto &verts = zstets->getParticles<true>();
    auto &eles = zstets->getQuadraturePoints();
    const tiles_t &coVerts =
        zsboundary ? zsboundary->getParticles() : tiles_t{};
    // auto totalEles = eles;
    // auto totalSurfEdges = eles;
    const tiles_t &coEles =
        zsboundary ? zsboundary->getQuadraturePoints() : tiles_t{};
    const tiles_t &coEdges =
        zsboundary ? (*zsboundary)[ZenoParticles::s_surfEdgeTag] : tiles_t{};

    auto coOffset = verts.size();

    static dtiles_t vtemp{
        verts.get_allocator(),
        {{"grad", 3},
         {"P", 9},
         // dirichlet boundary condition type; 0: NOT, 1: ZERO, 2: NONZERO
         {"BCorder", 1},
         {"BCbasis", 9},
         {"BCtarget", 3},
         {"BCfixed", 1},
         {"ws", 1}, // also as constraint jacobian
         {"cons", 3},
         {"lambda", 3},

         {"dir", 3},
         {"xn", 3},
         {"xn0", 3},
         {"xtilde", 3},
         {"temp", 3},
         {"r", 3},
         {"p", 3},
         {"q", 3}},
        verts.size() + coVerts.size()};
    static dtiles_t etemp{eles.get_allocator(), {{"He", 9 * 9}}, eles.size()};

    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    extForce = vec3{0, -9, 0};
    FEMSystem A{verts,  (*zstets)[ZenoParticles::s_surfEdgeTag],
                eles,   coVerts,
                coEles, vtemp,
                etemp,  dt,
                models};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    /// time integrator
    // set BC... info
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp),
             verts = proxy<space>({}, verts)] __device__(int i) mutable {
              vtemp("BCorder", i) = verts("BCorder", i);
              vtemp.template tuple<9>("BCbasis", i) =
                  verts.template pack<3, 3>("BCbasis", i);
              vtemp.template tuple<3>("BCtarget", i) =
                  verts.template pack<3>("BCtarget", i);
              vtemp("BCfixed", i) = verts("BCfixed", i);
            });
    cudaPol(zs::range(coVerts.size()), [vtemp = proxy<space>({}, vtemp),
                                        coverts = proxy<space>({}, coVerts),
                                        coOffset,
                                        dt] __device__(int i) mutable {
      auto x = coverts.pack<3>("x", i);
      auto v = coverts.pack<3>("v", i);
      vtemp("BCorder", coOffset + i) = 3;
      vtemp.template tuple<9>("BCbasis", coOffset + i) = mat3::identity();
      vtemp.template tuple<3>("BCtarget", coOffset + i) = x + v * dt;
      vtemp("BCfixed", coOffset + i) = v.l2NormSqr() == 0 ? 1 : 0;
    });
    // predict pos, initialize constrain weights
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt, extForce = extForce] __device__(int i) mutable {
              auto x = verts.pack<3>("x", i);
              auto v = verts.pack<3>("v", i);
              vtemp.tuple<3>("xtilde", i) = x + v * dt;
              vtemp("ws", i) = zs::sqrt(verts("m", i));
              vtemp.tuple<3>("lambda", i) = vec3::zeros();
              vtemp.tuple<3>("xn", i) = x;
              vtemp.tuple<3>("xt", i) = x;
            });
    cudaPol(zs::range(coVerts.size()),
            [vtemp = proxy<space>({}, vtemp),
             coverts = proxy<space>({}, coVerts), coOffset,
             dt] __device__(int i) mutable {
              auto x = coverts.pack<3>("x", i);
              auto v = coverts.pack<3>("v", i);
              vtemp.tuple<3>("xtilde", coOffset + i) = x + v * dt;
              vtemp("ws", coOffset + i) = zs::sqrt(coverts("m", i) * 1e3);
              vtemp.tuple<3>("lambda", coOffset + i) = vec3::zeros();
              vtemp.tuple<3>("xn", coOffset + i) = x;
              vtemp.tuple<3>("xt", coOffset + i) = x;
            });
    // fix initial x for all bcs if not feasible
    if constexpr (false) { // dont do this in augmented lagrangian
      cudaPol(zs::range(verts.size()),
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                auto x = verts.pack<3>("x", vi);
                if (int BCorder = vtemp("BCorder", vi); BCorder > 0) {
                  auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
                  auto BCtarget = vtemp.pack<3>("BCtarget", vi);
                  x = BCbasis.transpose() * x;
                  for (int d = 0; d != BCorder; ++d)
                    x[d] = BCtarget[d];
                  x = BCbasis * x;
                  verts.tuple<3>("x", vi) = x;
                }
                vtemp.tuple<3>("xn", vi) = x;
              });
      projectDBC = true;
      BCsatisfied = true;
    } else {
      projectDBC = false;
      BCsatisfied = false;
    }
    kappa = kappa0;

    /// optimizer
    for (int newtonIter = 0; newtonIter != 100; ++newtonIter) {
      // check constraints
      if (A.areConstraintsSatisfied(cudaPol)) {
        projectDBC = true;
        BCsatisfied = true;
      }
      A.findCollisionConstraints(cudaPol, dHat, xi);
      auto [npp, npe, npt, nee, ncspt, ncsee] = A.getCnts();
      // construct gradient, prepare hessian, prepare preconditioner
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P", i) = mat3::zeros();
              });
      cudaPol(zs::range(verts.size()),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               extForce = extForce, dt] __device__(int i) mutable {
                auto m = verts("m", i);
                auto v = verts.pack<3>("v", i);
                vtemp.tuple<3>("grad", i) =
                    m * extForce * dt * dt -
                    m * (vtemp.pack<3>("xn", i) - vtemp.pack<3>("xtilde", i));
              });
      cudaPol(zs::range(coVerts.size()),
              [vtemp = proxy<space>({}, vtemp),
               coverts = proxy<space>({}, coVerts), coOffset,
               dt] __device__(int i) mutable {
                auto m = zs::sqr(vtemp("ws", coOffset + i));
                auto v = coverts.pack<3>("v", i);
                // only inertial, no extforce grad
                vtemp.tuple<3>("grad", coOffset + i) =
                    -m * (vtemp.pack<3>("xn", coOffset + i) -
                          vtemp.pack<3>("xtilde", coOffset + i));
              });
      match([&](auto &elasticModel) {
        A.computeElasticGradientAndHessian(cudaPol, elasticModel);
      })(models.getElasticModel());
      A.computeBoundaryBarrierGradientAndHessian(cudaPol);
      A.computeBarrierGradientAndHessian(cudaPol);

      // rotate gradient and project
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               projectDBC = projectDBC] __device__(int i) mutable {
                auto grad = vtemp.pack<3, 3>("BCbasis", i).transpose() *
                            vtemp.pack<3>("grad", i);
                int BCfixed = vtemp("BCfixed", i);
                if (projectDBC || BCfixed == 1) {
                  if (int BCorder = vtemp("BCorder", i); BCorder > 0)
                    for (int d = 0; d != BCorder; ++d)
                      grad(d) = 0;
                }
                vtemp.tuple<3>("grad", i) = grad;
              });
      // apply constraints (augmented lagrangians) after rotation!
      if (!BCsatisfied) {
        // grad
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp),
                 kappa = kappa] __device__(int i) mutable {
                  // computed during the previous constraint residual check
                  auto cons = vtemp.pack<3>("cons", i);
                  auto w = vtemp("ws", i);
                  vtemp.tuple<3>("grad", i) =
                      vtemp.pack<3>("grad", i) -
                      (-w * vtemp.pack<3>("lambda", i) + kappa * w * cons);
                  for (int d = 0; d != 4; ++d)
                    if (cons[d] != 0) {
                      vtemp("P", 4 * d, i) += kappa * w;
                    }
                });
        // hess (embedded in multiply)
      }

      // prepare preconditioner
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, verts)] __device__(int i) mutable {
                auto m = verts("m", i);
                vtemp("P", 0, i) += m;
                vtemp("P", 4, i) += m;
                vtemp("P", 8, i) += m;
              });
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P", i) = inverse(vtemp.pack<3, 3>("P", i));
              });

      // modify initial x so that it satisfied the constraint.

      // A dir = grad
      {
        // solve for A dir = grad;
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("dir", i) = vec3::zeros();
                });
        // temp = A * dir
        A.multiply(cudaPol, "dir", "temp");
        // r = grad - temp
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("r", i) =
                      vtemp.pack<3>("grad", i) - vtemp.pack<3>("temp", i);
                });
        A.project(cudaPol, "r");
        A.precondition(cudaPol, "r", "q");
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("p", i) = vtemp.pack<3>("q", i);
                });
        T zTrk = dot(cudaPol, vtemp, "r", "q");
        auto residualPreconditionedNorm = std::sqrt(zTrk);
        auto localTol = 0.5 * residualPreconditionedNorm;
        int iter = 0;
        for (; iter != 10000; ++iter) {
          if (iter % 10 == 0)
            fmt::print("cg iter: {}, norm: {} (zTrk: {}) npp: {}, npe: {}, "
                       "npt: {}, nee: {}, ncspt: {}, ncsee: {}\n",
                       iter, residualPreconditionedNorm, zTrk, npp, npe, npt,
                       nee, ncspt, ncsee);
          if (zTrk < 0) {
            puts("what the heck?");
            getchar();
          }
          if (residualPreconditionedNorm <= localTol)
            break;
          A.multiply(cudaPol, "p", "temp");
          A.project(cudaPol, "temp");

          T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");
          cudaPol(range(verts.size()), [vtemp = proxy<space>({}, vtemp),
                                        alpha] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple<3>("dir", vi) =
                vtemp.pack<3>("dir", vi) + alpha * vtemp.pack<3>("p", vi);
            vtemp.tuple<3>("r", vi) =
                vtemp.pack<3>("r", vi) - alpha * vtemp.pack<3>("temp", vi);
          });

          A.precondition(cudaPol, "r", "q");
          auto zTrkLast = zTrk;
          zTrk = dot(cudaPol, vtemp, "q", "r");
          auto beta = zTrk / zTrkLast;
          cudaPol(range(verts.size()), [vtemp = proxy<space>({}, vtemp),
                                        beta] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple<3>("p", vi) =
                vtemp.pack<3>("q", vi) + beta * vtemp.pack<3>("p", vi);
          });

          residualPreconditionedNorm = std::sqrt(zTrk);
        } // end cg step
      }
      // recover rotated solution
      cudaPol(Collapse{vtemp.size()},
              [vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
                vtemp.tuple<3>("dir", vi) =
                    vtemp.pack<3, 3>("BCbasis", vi) * vtemp.pack<3>("dir", vi);
              });
      // check "dir" inf norm
      T res = infNorm(cudaPol, vtemp, "dir") / dt;
      T cons_res = A.constraintResidual(cudaPol);
      if (res < 1e-6 && cons_res == 0) {
        fmt::print("\t# newton optimizer ends in {} iters with residual {}\n",
                   newtonIter, res);
        break;
      }

      fmt::print("newton iter {}: direction residual {}, grad residual {}\n",
                 newtonIter, res, infNorm(cudaPol, vtemp, "grad"));

      // xn0 <- xn for line search
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0{};
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel, "xn0", !BCsatisfied);
      })(models.getElasticModel());

      // line search
      T alpha = 1.;
      find_ground_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha);
      fmt::print("\tstepsize after ground: {}\n", alpha);
      A.intersectionFreeStepsize(cudaPol, xi, alpha);
      fmt::print("\tstepsize after intersection-free: {}\n", alpha);
      A.findCCDConstraints(cudaPol, alpha, xi);
      A.intersectionFreeStepsize(cudaPol, xi, alpha);
      fmt::print("\tstepsize after ccd: {}\n", alpha);

      T E{E0};
#if 1
      do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });

        A.findCollisionConstraints(cudaPol, dHat, xi);
        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn", !BCsatisfied);
        })(models.getElasticModel());

        fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
        if (E < E0)
          break;

        alpha /= 2;
      } while (true);
#endif
      cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                        alpha] __device__(int i) mutable {
        vtemp.tuple<3>("xn", i) =
            vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
      });

      // update rule
      cons_res = A.constraintResidual(cudaPol);
      if (res < updateZoneTol && cons_res > consTol) {
        if (kappa < kappaMax)
          kappa *= 2;
        else {
          cudaPol(Collapse{vtemp.size()},
                  [vtemp = proxy<space>({}, vtemp),
                   kappa = kappa] __device__(int vi) mutable {
                    vtemp.tuple<3>("lambda", vi) =
                        vtemp.pack<3>("lambda", vi) -
                        kappa * vtemp("ws", vi) * vtemp.pack<3>("cons", vi);
                  });
        }
      }
    } // end newton step

    // update velocity and positions
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt] __device__(int vi) mutable {
              auto newX = vtemp.pack<3>("xn", vi);
              verts.tuple<3>("x", vi) = newX;
              auto dv = (newX - vtemp.pack<3>("xtilde", vi)) / dt;
              auto vn = verts.pack<3>("v", vi);
              vn += dv;
              verts.tuple<3>("v", vi) = vn;
            });
    // not sure if this is necessary for numerical reasons
    if (coVerts.size())
      cudaPol(zs::range(coVerts.size()),
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, zsboundary->getParticles()), coOffset,
               dt] __device__(int vi) mutable {
                auto newX = vtemp.pack<3>("xn", coOffset + vi);
                verts.tuple<3>("x", vi) = newX;
                // no need to update v here. positions are moved accordingly
                // also, boundary velocies are set elsewhere
              });

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(CodimStepping,
           {{"ZSParticles", "ZSBoundaryPrimitives", {"float", "dt", "0.01"}},
            {"ZSParticles"},
            {},
            {"FEM"}});

} // namespace zeno

#include "Ipc.inl"