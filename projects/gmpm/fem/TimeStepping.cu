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
#include "zensim/types/Property.h"
#include <atomic>
#include <zeno/VDBGrid.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

struct ExplicitTimeStepping : INode {
  using dtiles_t = zs::TileVector<double, 32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<double, 3>;
  using mat3 = zs::vec<double, 3, 3>;

  template <typename Model>
  void computeElasticImpulse(zs::CudaExecutionPolicy &cudaPol,
                             const Model &model, const tiles_t &verts,
                             const tiles_t &eles, dtiles_t &vtemp, float dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(zs::range(eles.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), model,
             dt] __device__(int ei) mutable {
              auto DmInv = eles.pack<3, 3>("IB", ei);
              auto dFdX = dFdXMatrix(DmInv);
              auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
              vec3 xs[4] = {verts.pack<3>("x", inds[0]).cast<double>(),
                            verts.pack<3>("x", inds[1]).cast<double>(),
                            verts.pack<3>("x", inds[2]).cast<double>(),
                            verts.pack<3>("x", inds[3]).cast<double>()};
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
              auto vfdt = -vole * dt * (dFdXT * vecP);

              for (int i = 0; i != 4; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                  atomic_add(exec_cuda, &vtemp("grad", d, vi), vfdt(i * 3 + d));
              }
            });
  }

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");
    auto &verts = zstets->getParticles();
    auto &eles = zstets->getQuadraturePoints();

    dtiles_t vtemp{verts.get_allocator(), {{"grad", 3}}, verts.size()};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt] __device__(int i) mutable {
              // vtemp.tuple<3>("grad", i) = vec3::zeros();
              // gravity impulse
              auto m = verts("m", i);
              vtemp.tuple<3>("grad", i) = m * vec3{0, -9, 0} * dt;
            });
    match([&](auto &elasticModel) {
      computeElasticImpulse(cudaPol, elasticModel, verts, eles, vtemp, dt);
    })(models.getElasticModel());

    // projection
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), dt] __device__(int vi) mutable {
              if (verts("x", 1, vi) > 0.5)
                vtemp.tuple<3>("grad", vi) = vec3::zeros();
            });

    // update velocity and positions
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             eles = proxy<space>({}, eles), dt] __device__(int vi) mutable {
              auto fdt = vtemp.pack<3>("grad", vi);
              auto m = verts("m", vi);
              auto dv = fdt / m * dt;
              auto vn = verts.pack<3>("v", vi);
              vn += dv;
              verts.tuple<3>("v", vi) = vn;
              verts.tuple<3>("x", vi) = verts.pack<3>("x", vi) + vn * dt;
            });

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ExplicitTimeStepping, {{"ZSParticles", {"float", "dt", "0.01"}},
                                  {"ZSParticles"},
                                  {},
                                  {"FEM"}});

struct ImplicitTimeStepping : INode {
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

  /// ref: codim-ipc
  void precompute_constraints(zs::CudaExecutionPolicy &pol,
                              ZenoParticles &zstets, const dtiles_t &vtemp,
                              T dHat, T xi, zs::Vector<pair_t> &PP,
                              zs::Vector<T> &wPP, zs::Vector<int> &nPP,
                              zs::Vector<pair3_t> &PE, zs::Vector<T> &wPE,
                              zs::Vector<int> &nPE, zs::Vector<pair4_t> &PT,
                              zs::Vector<T> &wPT, zs::Vector<int> &nPT,
                              zs::Vector<pair4_t> &EE, zs::Vector<T> &wEE,
                              zs::Vector<int> &nEE) {
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

    /// tri
    const auto &surfaces = zstets[ZenoParticles::s_surfTriTag];
    {
      auto bvs = retrieve_bounding_volumes(pol, vtemp, surfaces, wrapv<3>{},
                                           0.f, "xn");
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
                                           0.f, "xn");
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
         wPT = proxy<space>(wPT), nPT = proxy<space>(nPT),
         thickness = dHat + xi, xi2, activeGap2] ZS_LAMBDA(int svi) mutable {
          auto vi = reinterpret_bits<int>(svs("inds", svi));
          auto p = vtemp.template pack<3>("xn", vi);
          auto wp = svs("w", vi) / 4;
          auto [mi, ma] = get_bounding_box(p - thickness, p + thickness);
          auto bv = bv_t{mi, ma};
          bvh.iter_neighbors(bv, [&](int stI) {
            auto tri = sts.template pack<3>("inds", stI)
                           .template reinterpret_bits<int>();
            if (vi == tri[0] || vi == tri[1] || vi == tri[2])
              return;
            // all affected by sticky boundary conditions
            if (reinterpret_bits<int>(verts("BCorder", vi)) == 3 &&
                reinterpret_bits<int>(verts("BCorder", tri[0])) == 3 &&
                reinterpret_bits<int>(verts("BCorder", tri[1])) == 3 &&
                reinterpret_bits<int>(verts("BCorder", tri[2])) == 3)
              return;
            // ccd
            auto t0 = vtemp.template pack<3>("xn", tri[0]);
            auto t1 = vtemp.template pack<3>("xn", tri[1]);
            auto t2 = vtemp.template pack<3>("xn", tri[2]);
            if (!pt_cd_broadphase(p, t0, t1, t2, thickness))
              return;
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
         nEE = proxy<space>(nEE), thickness = dHat + xi, xi2,
         activeGap2] ZS_LAMBDA(int sei) mutable {
          auto eiInds = ses.template pack<2>("inds", sei)
                            .template reinterpret_bits<int>();
          auto selfWe = ses("w", sei);
          bool selfFixed =
              reinterpret_bits<int>(verts("BCorder", eiInds[0])) == 3 &&
              reinterpret_bits<int>(verts("BCorder", eiInds[1])) == 3;
          auto x0 = vtemp.template pack<3>("xn", eiInds[0]);
          auto x1 = vtemp.template pack<3>("xn", eiInds[1]);
          auto [mi, ma] = get_bounding_box(x0, x1);
          auto bv = bv_t{mi - thickness, ma + thickness};
          bvh.iter_neighbors(bv, [&](int sej) {
            // if (sei > sej) return;
            auto ejInds = ses.template pack<2>("inds", sej)
                              .template reinterpret_bits<int>();
            if (eiInds[0] == ejInds[0] || eiInds[0] == ejInds[1] ||
                eiInds[1] == ejInds[0] || eiInds[1] == ejInds[1])
              return;
            // all affected by sticky boundary conditions
            if (selfFixed &&
                reinterpret_bits<int>(verts("BCorder", ejInds[0])) == 3 &&
                reinterpret_bits<int>(verts("BCorder", ejInds[1])) == 3)
              return;
            // ccd
            auto eb0 = vtemp.template pack<3>("xn", ejInds[0]);
            auto eb1 = vtemp.template pack<3>("xn", ejInds[1]);
            if (!ee_cd_broadphase(x0, x1, eb0, eb1, thickness))
              return;
            auto we = (selfWe + ses("w", sej)) / 4;
            switch (ee_distance_type(x0, x1, eb0, eb1)) {
            case 0: {
              if (dist2_pp(x0, eb0) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[0], ejInds[0]};
                wPP[no] = we;
              }
              break;
            }
            case 1: {
              if (dist2_pp(x0, eb1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[0], ejInds[1]};
                wPP[no] = we;
              }
              break;
            }
            case 2: {
              if (dist2_pe(x0, eb0, eb1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{eiInds[0], ejInds[0], ejInds[1]};
                wPE[no] = we;
              }
              break;
            }
            case 3: {
              if (dist2_pp(x1, eb0) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[1], ejInds[0]};
                wPP[no] = we;
              }
              break;
            }
            case 4: {
              if (dist2_pp(x1, eb1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPP[0], 1);
                PP[no] = pair_t{eiInds[1], ejInds[1]};
                wPP[no] = we;
              }
              break;
            }
            case 5: {
              if (dist2_pe(x1, eb0, eb1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{eiInds[1], ejInds[0], ejInds[1]};
                wPE[no] = we;
              }
              break;
            }
            case 6: {
              if (dist2_pe(eb0, x0, x1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{ejInds[0], eiInds[0], eiInds[1]};
                wPE[no] = we;
              }
              break;
            }
            case 7: {
              if (dist2_pe(eb1, x0, x1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nPE[0], 1);
                PE[no] = pair3_t{ejInds[1], eiInds[0], eiInds[1]};
                wPE[no] = we;
              }
              break;
            }
            case 8: {
              if (dist2_ee(x0, x1, eb0, eb1) - xi2 < activeGap2) {
                auto no = atomic_add(exec_cuda, &nEE[0], 1);
                EE[no] = pair4_t{eiInds[0], eiInds[1], ejInds[0], ejInds[1]};
                wEE[no] = we;
              }
              break;
            }
            default:
              break;
            }
          });
        });
    fmt::print("contact indeed detected. nPP: {}, nPE: {}, nPT: {}, nEE: {}\n",
               nPP.getVal(), nPE.getVal(), nPT.getVal(), nEE.getVal());
    if (nPP.getVal() > 0 || nPE.getVal() > 0 || nPT.getVal() > 0 ||
        nEE.getVal() > 0) {
      // getchar();
    }
  }
  void computeInversionFreeStepSize(zs::CudaExecutionPolicy &pol,
                                    const tiles_t &eles, const dtiles_t &vtemp,
                                    T &stepSize) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    zs::Vector<T> stepSizes{eles.get_allocator(), eles.size()},
        minSize{eles.get_allocator(), 1};
    minSize.setVal(stepSize);
    pol(zs::Collapse{eles.size()},
        [eles = proxy<space>({}, eles), minSize = proxy<space>(minSize),
         vtemp = proxy<space>({}, vtemp), stepSize] __device__(int ei) mutable {
          auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
          T x1 = vtemp("xn", 0, inds[0]);
          T x2 = vtemp("xn", 0, inds[1]);
          T x3 = vtemp("xn", 0, inds[2]);
          T x4 = vtemp("xn", 0, inds[3]);

          T y1 = vtemp("xn", 1, inds[0]);
          T y2 = vtemp("xn", 1, inds[1]);
          T y3 = vtemp("xn", 1, inds[2]);
          T y4 = vtemp("xn", 1, inds[3]);

          T z1 = vtemp("xn", 2, inds[0]);
          T z2 = vtemp("xn", 2, inds[1]);
          T z3 = vtemp("xn", 2, inds[2]);
          T z4 = vtemp("xn", 2, inds[3]);

          T p1 = vtemp("dir", 0, inds[0]);
          T p2 = vtemp("dir", 0, inds[1]);
          T p3 = vtemp("dir", 0, inds[2]);
          T p4 = vtemp("dir", 0, inds[3]);

          T q1 = vtemp("dir", 1, inds[0]);
          T q2 = vtemp("dir", 1, inds[1]);
          T q3 = vtemp("dir", 1, inds[2]);
          T q4 = vtemp("dir", 1, inds[3]);

          T r1 = vtemp("dir", 2, inds[0]);
          T r2 = vtemp("dir", 2, inds[1]);
          T r3 = vtemp("dir", 2, inds[2]);
          T r4 = vtemp("dir", 2, inds[3]);

          T a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 -
                r1 * p2 * q3 + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 -
                q1 * p2 * r4 + q1 * r2 * p4 + r1 * p2 * q4 - r1 * q2 * p4 -
                p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4 - q1 * r3 * p4 -
                r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4 -
                q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
          T b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 -
                z1 * p2 * q3 + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 -
                y2 * p1 * r3 + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 -
                x3 * q1 * r2 + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 -
                z3 * p1 * q2 + z3 * q1 * p2 + x1 * q2 * r4 - x1 * r2 * q4 -
                y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4 - z1 * q2 * p4 -
                x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4 -
                z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 -
                y4 * p1 * r2 + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 -
                x1 * q3 * r4 + x1 * r3 * q4 + y1 * p3 * r4 - y1 * r3 * p4 -
                z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4 - x3 * r1 * q4 -
                y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4 -
                x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 -
                z4 * p1 * q3 + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 -
                y2 * p3 * r4 + y2 * r3 * p4 + z2 * p3 * q4 - z2 * q3 * p4 -
                x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4 - y3 * r2 * p4 -
                z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3 -
                y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
          T c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 +
                y1 * x2 * r3 - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 -
                z1 * x2 * q3 + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 -
                x2 * y3 * r1 + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 -
                z2 * x3 * q1 + z2 * y3 * p1 + x1 * y2 * r4 - x1 * z2 * q4 -
                x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4 + y1 * z2 * p4 +
                y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4 -
                z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 -
                y2 * x4 * r1 + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 -
                x1 * y3 * r4 + x1 * z3 * q4 + x1 * y4 * r3 - x1 * z4 * q3 +
                y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3 + y1 * z4 * p3 -
                z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3 -
                x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 -
                z3 * x4 * q1 + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 -
                x2 * y4 * r3 + x2 * z4 * q3 - y2 * x3 * r4 + y2 * z3 * p4 +
                y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4 - z2 * y3 * p4 -
                z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2 -
                y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
          T d = ((T)1.0 - (T)0.2) *
                (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 -
                 z1 * x2 * y3 + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 -
                 y1 * x2 * z4 + y1 * z2 * x4 + z1 * x2 * y4 - z1 * y2 * x4 -
                 x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4 - y1 * z3 * x4 -
                 z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4 -
                 y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);

          T t = zs::math::get_smallest_positive_real_cubic_root(a, b, c, d,
                                                                (T)1.e-6);
#if 0
      if (t >= 0)
        stepSizes[ei] = t;
      else
        stepSizes[ei] = limits<T>::max();
#else
      if (t < stepSize && t >= 0)
        atomic_min(exec_cuda, &minSize[0], t);
#endif
        });

#if 0
    zs::Vector<T> res{eles.get_allocator(), 1};
    zs::reduce(pol, std::begin(stepSizes), std::end(stepSizes), std::begin(res),
               stepSize, zs::getmin<T>{});
    stepSize = res.getVal();
#else
    stepSize = minSize.getVal();
#endif
    fmt::print("inversion free alpha: {}\n", stepSize);
  }
  void find_ground_intersection_free_stepsize(zs::CudaExecutionPolicy &pol,
                                              const ZenoParticles &zstets,
                                              const dtiles_t &vtemp,
                                              T &stepSize) {
    using namespace zs;
    constexpr T slackness = 0.8;
    constexpr auto space = execspace_e::cuda;

    const auto &verts = zstets.getParticles();
    const auto &surfVerts = zstets[ZenoParticles::s_surfVertTag];

    ///
    // query pt
    zs::Vector<T> finalAlpha{surfVerts.get_allocator(), 1};
    finalAlpha.setVal(stepSize);
    pol(Collapse{surfVerts.size()},
        [svs = proxy<space>({}, surfVerts), vtemp = proxy<space>({}, vtemp),
         verts = proxy<space>({}, verts),
         // boundary
         gn = s_groundNormal, finalAlpha = proxy<space>(finalAlpha),
         stepSize] ZS_LAMBDA(int svi) mutable {
          auto vi = reinterpret_bits<int>(svs("inds", svi));
          // this vert affected by sticky boundary conditions
          if (reinterpret_bits<int>(verts("BCorder", vi)) == 3)
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
    fmt::print("ground alpha: {}\n", stepSize);
  }

  struct FEMSystem {
    void computeBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
      using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
      using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
      using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
      auto numPP = nPP.getVal();
      pol(range(numPP), [vtemp = proxy<space>({}, vtemp),
                         tempPP = proxy<space>({}, tempPP),
                         PP = proxy<space>(PP), wPP = proxy<space>(wPP),
                         xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int ppi) mutable {
        auto pp = PP[ppi];
        auto x0 = vtemp.pack<3>("xn", pp[0]);
        auto x1 = vtemp.pack<3>("xn", pp[1]);
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
            atomic_add(exec_cuda, &vtemp("grad", d, pp[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pp[1]), grad(1, d));
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
          tempPP.tuple<36>("H", ppi) = ppHess;
        } else {
          using Mat6x6 = zs::vec<T, 6, 6>;
          tempPP.tuple<36>("H", ppi) = Mat6x6::zeros();
        }
      });
      auto numPE = nPE.getVal();
      pol(range(numPE), [vtemp = proxy<space>({}, vtemp),
                         tempPE = proxy<space>({}, tempPE),
                         PE = proxy<space>(PE), wPE = proxy<space>(wPE),
                         xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int pei) mutable {
        auto pe = PE[pei];
        auto p = vtemp.pack<3>("xn", pe[0]);
        auto e0 = vtemp.pack<3>("xn", pe[1]);
        auto e1 = vtemp.pack<3>("xn", pe[2]);

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
            atomic_add(exec_cuda, &vtemp("grad", d, pe[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pe[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pe[2]), grad(2, d));
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
          tempPE.tuple<81>("H", pei) = peHess;
        } else {
          using Mat9x9 = zs::vec<T, 9, 9>;
          tempPE.tuple<81>("H", pei) = Mat9x9::zeros();
        }
      });
      auto numPT = nPT.getVal();
      pol(range(numPT), [vtemp = proxy<space>({}, vtemp),
                         tempPT = proxy<space>({}, tempPT),
                         PT = proxy<space>(PT), wPT = proxy<space>(wPT),
                         xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int pti) mutable {
        auto pt = PT[pti];
        auto p = vtemp.pack<3>("xn", pt[0]);
        auto t0 = vtemp.pack<3>("xn", pt[1]);
        auto t1 = vtemp.pack<3>("xn", pt[2]);
        auto t2 = vtemp.pack<3>("xn", pt[3]);

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
            atomic_add(exec_cuda, &vtemp("grad", d, pt[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pt[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pt[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp("grad", d, pt[3]), grad(3, d));
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
          tempPT.tuple<144>("H", pti) = ptHess;
        } else {
          using Mat12x12 = zs::vec<T, 12, 12>;
          tempPT.tuple<144>("H", pti) = Mat12x12::zeros();
        }
      });
      auto numEE = nEE.getVal();
      pol(range(numEE), [vtemp = proxy<space>({}, vtemp),
                         tempEE = proxy<space>({}, tempEE),
                         EE = proxy<space>(EE), wEE = proxy<space>(wEE),
                         xi2 = xi * xi, dHat = dHat, activeGap2,
                         kappa = kappa] __device__(int eei) mutable {
        auto ee = EE[eei];
        auto ea0 = vtemp.pack<3>("xn", ee[0]);
        auto ea1 = vtemp.pack<3>("xn", ee[1]);
        auto eb0 = vtemp.pack<3>("xn", ee[2]);
        auto eb1 = vtemp.pack<3>("xn", ee[3]);

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
            atomic_add(exec_cuda, &vtemp("grad", d, ee[0]), grad(0, d));
            atomic_add(exec_cuda, &vtemp("grad", d, ee[1]), grad(1, d));
            atomic_add(exec_cuda, &vtemp("grad", d, ee[2]), grad(2, d));
            atomic_add(exec_cuda, &vtemp("grad", d, ee[3]), grad(3, d));
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
          tempEE.tuple<144>("H", eei) = eeHess;
        } else {
          using Mat12x12 = zs::vec<T, 12, 12>;
          tempEE.tuple<144>("H", eei) = Mat12x12::zeros();
        }
      });
      return;
    }
    void
    computeBoundaryBarrierGradientAndHessian(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp), tempPB = proxy<space>({}, tempPB),
           gn = s_groundNormal, dHat2 = dHat * dHat,
           kappa = kappa] ZS_LAMBDA(int vi) mutable {
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
                atomic_add(exec_cuda, &vtemp("grad", d, vi), grad(d));
            }

            auto param = 4 * H_b * dist2 + 2 * g_b;
            auto hess = mat3::zeros();
            if (dist2 < dHat2 && param > 0) {
              auto nn = dyadic_prod(gn, gn);
              hess = (kappa * param) * nn;
            }
            tempPB.tuple<9>("H", vi) = hess;
          });
      return;
    }
    template <typename Pol, typename Model>
    T energy(Pol &pol, const Model &model, const zs::SmallString tag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
      pol(range(vtemp.size()), [verts = proxy<space>({}, verts),
                                vtemp = proxy<space>({}, vtemp),
                                res = proxy<space>(res), tag,
                                dt = this->dt] __device__(int vi) mutable {
        // inertia
        auto m = verts("m", vi);
        auto x = vtemp.pack<3>(tag, vi);
        atomic_add(exec_cuda, &res[0],
                   (T)0.5 * m * (x - vtemp.pack<3>("xtilde", vi)).l2NormSqr());
        // gravity
        atomic_add(exec_cuda, &res[0],
                   -m * vec3{0, -9, 0}.dot(x - verts.pack<3>("x", vi)) * dt *
                       dt);
      });
      // elasticity
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model,
                               dt = this->dt] __device__(int ei) mutable {
        auto DmInv = eles.template pack<3, 3>("IB", ei);
        auto inds =
            eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
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
      // contacts
      {
        auto activeGap2 = dHat * dHat + 2 * xi * dHat;
        auto numPP = nPP.getVal();
        pol(range(numPP),
            [vtemp = proxy<space>({}, vtemp), tempPP = proxy<space>({}, tempPP),
             PP = proxy<space>(PP), wPP = proxy<space>(wPP),
             res = proxy<space>(res), xi2 = xi * xi, dHat = dHat, activeGap2,
             kappa = kappa] __device__(int ppi) mutable {
              auto pp = PP[ppi];
              auto x0 = vtemp.pack<3>("xn", pp[0]);
              auto x1 = vtemp.pack<3>("xn", pp[1]);
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
             res = proxy<space>(res), xi2 = xi * xi, dHat = dHat, activeGap2,
             kappa = kappa] __device__(int pei) mutable {
              auto pe = PE[pei];
              auto p = vtemp.pack<3>("xn", pe[0]);
              auto e0 = vtemp.pack<3>("xn", pe[1]);
              auto e1 = vtemp.pack<3>("xn", pe[2]);

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
             res = proxy<space>(res), xi2 = xi * xi, dHat = dHat, activeGap2,
             kappa = kappa] __device__(int pti) mutable {
              auto pt = PT[pti];
              auto p = vtemp.pack<3>("xn", pt[0]);
              auto t0 = vtemp.pack<3>("xn", pt[1]);
              auto t1 = vtemp.pack<3>("xn", pt[2]);
              auto t2 = vtemp.pack<3>("xn", pt[3]);

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
             res = proxy<space>(res), xi2 = xi * xi, dHat = dHat, activeGap2,
             kappa = kappa] __device__(int eei) mutable {
              auto ee = EE[eei];
              auto ea0 = vtemp.pack<3>("xn", ee[0]);
              auto ea1 = vtemp.pack<3>("xn", ee[1]);
              auto eb0 = vtemp.pack<3>("xn", ee[2]);
              auto eb1 = vtemp.pack<3>("xn", ee[3]);

              auto dist2 = dist2_ee(ea0, ea1, eb0, eb1);
              if (dist2 < xi2)
                printf("dist already smaller than xi!\n");
              if (dist2 - xi2 < activeGap2)
                atomic_add(exec_cuda, &res[0],
                           wEE[eei] * dHat *
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
      return res.getVal();
    }
    template <typename Pol> void project(Pol &pol, const zs::SmallString tag) {
#if 0
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // projection
      pol(zs::range(verts.size()),
          [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
           tag] ZS_LAMBDA(int vi) mutable {
            if (verts("x", 1, vi) > 0.5)
              vtemp.tuple<3>(tag, vi) = vec3::zeros();
          });
#endif
    }
    template <typename Pol>
    void precondition(Pol &pol, const zs::SmallString srcTag,
                      const zs::SmallString dstTag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // precondition
      pol(zs::range(verts.size()),
          [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
           srcTag, dstTag] ZS_LAMBDA(int vi) mutable {
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
        auto BCbasis = verts.template pack<3, 3>("BCbasis", vi);
        dx = BCbasis.transpose() * m * BCbasis * dx;
        for (int d = 0; d != 3; ++d)
          atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
      });
      // elasticity
      pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                           vtemp = proxy<space>({}, vtemp),
                           eles = proxy<space>({}, eles), dxTag,
                           bTag] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = 3;
        constexpr auto dimp1 = dim + 1;
        auto inds = eles.template pack<dimp1>("inds", ei)
                        .template reinterpret_bits<int>();
        zs::vec<T, dimp1 * dim> temp{};
        for (int vi = 0; vi != dimp1; ++vi)
          for (int d = 0; d != dim; ++d) {
            temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
          }
        auto He = etemp.template pack<dim * dimp1, dim * dimp1>("He", ei);

        temp = He * temp;

        for (int vi = 0; vi != dimp1; ++vi)
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
    }

    FEMSystem(const tiles_t &verts, const tiles_t &eles, dtiles_t &vtemp,
              dtiles_t &etemp, const zs::Vector<pair_t> &PP,
              const zs::Vector<T> &wPP, const zs::Vector<int> &nPP,
              const zs::Vector<pair3_t> &PE, const zs::Vector<T> &wPE,
              const zs::Vector<int> &nPE, const zs::Vector<pair4_t> &PT,
              const zs::Vector<T> &wPT, const zs::Vector<int> &nPT,
              const zs::Vector<pair4_t> &EE, const zs::Vector<T> &wEE,
              const zs::Vector<int> &nEE, T dHat, T xi, T dt)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp}, PP{PP},
          wPP{wPP}, nPP{nPP},
          tempPP{PP.get_allocator(), {{"H", 36}}, PP.size()}, PE{PE}, wPE{wPE},
          nPE{nPE}, tempPE{PE.get_allocator(), {{"H", 81}}, PE.size()}, PT{PT},
          wPT{wPT}, nPT{nPT},
          tempPT{PT.get_allocator(), {{"H", 144}}, PT.size()}, EE{EE}, wEE{wEE},
          nEE{nEE}, tempEE{EE.get_allocator(), {{"H", 144}}, EE.size()},
          tempPB{verts.get_allocator(), {{"H", 9}}, verts.size()}, dHat{dHat},
          xi{xi}, dt{dt} {}

    const tiles_t &verts;
    const tiles_t &eles;
    dtiles_t &vtemp;
    dtiles_t &etemp;
    // contacts
    const zs::Vector<pair_t> &PP;
    const zs::Vector<T> &wPP;
    const zs::Vector<int> &nPP;
    dtiles_t tempPP;
    const zs::Vector<pair3_t> &PE;
    const zs::Vector<T> &wPE;
    const zs::Vector<int> &nPE;
    dtiles_t tempPE;
    const zs::Vector<pair4_t> &PT;
    const zs::Vector<T> &wPT;
    const zs::Vector<int> &nPT;
    dtiles_t tempPT;
    const zs::Vector<pair4_t> &EE;
    const zs::Vector<T> &wEE;
    const zs::Vector<int> &nEE;
    dtiles_t tempEE;
    // boundary contacts
    dtiles_t tempPB;
    // end contacts
    T dHat, xi, dt;
  };

  template <typename Model>
  void computeElasticGradientAndHessian(zs::CudaExecutionPolicy &cudaPol,
                                        const Model &model,
                                        const tiles_t &verts,
                                        const tiles_t &eles, dtiles_t &vtemp,
                                        dtiles_t &etemp, float dt) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                     etemp = proxy<space>({}, etemp),
                                     verts = proxy<space>({}, verts),
                                     eles = proxy<space>({}, eles), model,
                                     dt] __device__(int ei) mutable {
      auto DmInv = eles.pack<3, 3>("IB", ei);
      auto dFdX = dFdXMatrix(DmInv);
      auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
      vec3 xs[4] = {vtemp.pack<3>("xn", inds[0]), vtemp.pack<3>("xn", inds[1]),
                    vtemp.pack<3>("xn", inds[2]), vtemp.pack<3>("xn", inds[3])};
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
          atomic_add(exec_cuda, &vtemp("grad", d, vi), vfdt2(i * 3 + d));
      }

      // hessian rotation: trans^T hess * trans
      // left trans^T: multiplied on rows
      // right trans: multiplied on cols
      mat3 BCbasis[4];
      int BCorder[4];
      for (int i = 0; i != 4; ++i) {
        BCbasis[i] = verts.pack<3, 3>("BCbasis", inds[i]);
        BCorder[i] = reinterpret_bits<int>(verts("BCorder", inds[i]));
      }
      auto Hq = model.first_piola_derivative(F, true_c);
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
      etemp.tuple<12 * 12>("He", ei) = H;
    });
  }

  T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
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
  T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
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

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    std::shared_ptr<ZenoParticles> zsboundary;
    if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
      zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");
    auto &verts = zstets->getParticles();
    auto &eles = zstets->getQuadraturePoints();

    static dtiles_t vtemp{verts.get_allocator(),
                          {{"grad", 3},
                           {"P", 9},
                           {"dir", 3},
                           {"xn", 3},
                           {"xn0", 3},
                           {"xtilde", 3},
                           {"temp", 3},
                           {"r", 3},
                           {"p", 3},
                           {"q", 3}},
                          verts.size()};
    static dtiles_t etemp{eles.get_allocator(), {{"He", 12 * 12}}, eles.size()};
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

    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{verts, eles, vtemp, etemp, PP,  wPP, nPP,  PE, wPE, nPE,
                PT,    wPT,  nPT,   EE,    wEE, nEE, dHat, xi, dt};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    /// time integrator
    // predict pos
    cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
             dt] __device__(int i) mutable {
              auto x = verts.pack<3>("x", i);
              auto v = verts.pack<3>("v", i);
              vtemp.tuple<3>("xtilde", i) = x + v * dt;
            });
    // fix initial x for all bcs if not feasible
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp),
             verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto x = verts.pack<3>("x", vi);
              if (auto BCorder = reinterpret_bits<int>(verts("BCorder", vi));
                  BCorder > 0) {
                auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
                auto BCtarget = verts.pack<3>("BCtarget", vi);
                x = BCbasis.transpose() * x;
                for (int d = 0; d != BCorder; ++d)
                  x[d] = BCtarget[d];
                x = BCbasis * x;
                verts.tuple<3>("x", vi) = x;
              }
              vtemp.tuple<3>("xn", vi) = x;
            });

    precompute_constraints(cudaPol, *zstets, vtemp, dHat, xi, PP, wPP, nPP, PE,
                           wPE, nPE, PT, wPT, nPT, EE, wEE, nEE);

    /// optimizer
    for (int newtonIter = 0; newtonIter != 100; ++newtonIter) {
      // construct gradient, prepare hessian
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               dt] __device__(int i) mutable {
                auto m = verts("m", i);
                auto v = verts.pack<3>("v", i);
                vtemp.tuple<3>("grad", i) =
                    m * vec3{0, -9, 0} * dt * dt -
                    m * (vtemp.pack<3>("xn", i) - vtemp.pack<3>("xtilde", i));
              });
      match([&](auto &elasticModel) {
        computeElasticGradientAndHessian(cudaPol, elasticModel, verts, eles,
                                         vtemp, etemp, dt);
      })(models.getElasticModel());
      A.computeBarrierGradientAndHessian(cudaPol);
      A.computeBoundaryBarrierGradientAndHessian(cudaPol);

      // rotate gradient and project
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               dt] __device__(int i) mutable {
                auto grad = verts.pack<3, 3>("BCbasis", i).transpose() *
                            vtemp.pack<3>("grad", i);
                if (auto BCorder = reinterpret_bits<int>(verts("BCorder", i));
                    BCorder > 0)
                  for (int d = 0; d != BCorder; ++d)
                    grad(d) = 0;
                vtemp.tuple<3>("grad", i) = grad;
              });

      // prepare preconditioner
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, verts)] __device__(int i) mutable {
                auto m = verts("m", i);
                vtemp.tuple<9>("P", i) = mat3::zeros();
                vtemp("P", 0, i) = m;
                vtemp("P", 4, i) = m;
                vtemp("P", 8, i) = m;
              });
      cudaPol(zs::range(eles.size()),
              [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, etemp),
               eles = proxy<space>({}, eles)] __device__(int ei) mutable {
                constexpr int dim = 3;
                constexpr auto dimp1 = dim + 1;
                auto inds =
                    eles.pack<dimp1>("inds", ei).reinterpret_bits<int>();
                auto He = etemp.pack<dim * dimp1, dim * dimp1>("He", ei);
                for (int vi = 0; vi != dimp1; ++vi) {
                  for (int i = 0; i != dim; ++i)
                    for (int j = 0; j != dim; ++j) {
                      atomic_add(exec_cuda, &vtemp("P", i * dim + j, inds[vi]),
                                 He(vi * dim + i, vi * dim + j));
                    }
                }
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
        auto localTol = std::min(0.01 * residualPreconditionedNorm, 1e-7);
        int iter = 0;
        for (; iter != 1000; ++iter) {
          if (iter % 10 == 0)
            fmt::print("cg iter: {}, norm: {}\n", iter,
                       residualPreconditionedNorm);
          if (residualPreconditionedNorm <= localTol)
            break;
          A.multiply(cudaPol, "p", "temp");
          A.project(cudaPol, "temp");

          T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");
          cudaPol(range(verts.size()), [verts = proxy<space>({}, verts),
                                        vtemp = proxy<space>({}, vtemp),
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
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                vtemp.tuple<3>("dir", vi) =
                    verts.pack<3, 3>("BCbasis", vi) * vtemp.pack<3>("dir", vi);
              });
      // check "dir" inf norm
      T res = infNorm(cudaPol, vtemp, "dir");
      if (res < 1e-6) {
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
        E0 = A.energy(cudaPol, elasticModel, "xn0");
      })(models.getElasticModel());

      // line search
      T alpha = 1.;
      // computeInversionFreeStepSize(cudaPol, eles, vtemp, alpha);
      find_ground_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha);
#if 1
      while (find_self_intersection_free_stepsize(cudaPol, *zstets, vtemp,
                                                  alpha, xi)) {
        alpha /= 2;
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });
      }
      fmt::print(fg(fmt::color::dark_cyan),
                 "discrete intersection-free alpha: {}\n", alpha);
#endif
      find_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha, xi);
      //
      if (zsboundary)
        find_boundary_intersection_free_stepsize(cudaPol, *zstets, vtemp,
                                                 *zsboundary, (T)dt, alpha, xi);

#if 1
      T E{E0};
      do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });

        //
        precompute_constraints(cudaPol, *zstets, vtemp, dHat, xi, PP, wPP, nPP,
                               PE, wPE, nPE, PT, wPT, nPT, EE, wEE, nEE);
        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn");
        })(models.getElasticModel());

        fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
        if (E < E0)
          break;

        alpha /= 2;
      } while (true);
#else
      cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                        alpha] __device__(int i) mutable {
        vtemp.tuple<3>("xn", i) =
            vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
      });
      //
      precompute_constraints(cudaPol, *zstets, vtemp, dHat, xi, PP, wPP, nPP,
                             PE, wPE, nPE, PT, wPT, nPT, EE, wEE, nEE);
#endif
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

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ImplicitTimeStepping,
           {{"ZSParticles", "ZSBoundaryPrimitives", {"float", "dt", "0.01"}},
            {"ZSParticles"},
            {},
            {"FEM"}});

} // namespace zeno