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

  static constexpr vec3 s_groundNormal{0, 1, 0};

  inline static T kappaMax = 1e8;
  inline static T kappaMin = 1e3;
  inline static T kappa = 1e4;
  inline static T xi = 0; // 1e-2; // 2e-3;
  inline static T dHat = 0.001;

  /// ref: codim-ipc
  static void
  find_ground_intersection_free_stepsize(zs::CudaExecutionPolicy &pol,
                                         const ZenoParticles &zstets,
                                         const dtiles_t &vtemp, T &stepSize) {
    using namespace zs;
    constexpr T slackness = 0.8;
    constexpr auto space = execspace_e::cuda;

    const auto &verts = zstets.getParticles();

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
    fmt::print(fg(fmt::color::dark_cyan), "ground alpha: {}\n", stepSize);
  }

  struct FEMSystem {
    void computeBoundaryBarrierGradientAndHessian(
        zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag = "grad") {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(range(vtemp.size()),
          [verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
           tempPB = proxy<space>({}, tempPB), gTag, gn = s_groundNormal,
           dHat2 = dHat * dHat, kappa = kappa] ZS_LAMBDA(int vi) mutable {
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
            // hessian rotation: trans^T hess * trans
            // left trans^T: multiplied on rows
            // right trans: multiplied on cols
            {
              auto tmp = hess;
              auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
              int BCorder = reinterpret_bits<int>(verts("BCorder", vi));
              // rotate
              tmp = BCbasis.transpose() * tmp * BCbasis;
              // project
              if (BCorder > 0) {
                for (int i = 0; i != BCorder; ++i)
                  for (int j = 0; j != BCorder; ++j)
                    tmp(i, j) = (i == j ? 1 : 0);
              }
              for (int i = 0; i != 3; ++i)
                for (int j = 0; j != 3; ++j)
                  hess(i, j) = tmp(i, j);
            }
            tempPB.tuple<9>("H", vi) = hess;
            for (int i = 0; i != 3; ++i)
              for (int j = 0; j != 3; ++j) {
                atomic_add(exec_cuda, &vtemp("P", i * 3 + j, vi), hess(i, j));
              }
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
        auto IB = eles.template pack<2, 2>("IB", ei);
        auto inds =
            eles.template pack<3>("inds", ei).template reinterpret_bits<int>();
        auto vole = eles("vol", ei);
        vec3 xs[3] = {vtemp.template pack<3>(tag, inds[0]),
                      vtemp.template pack<3>(tag, inds[1]),
                      vtemp.template pack<3>(tag, inds[2])};
        mat2 A{};
        T E;
        {
          auto x1x0 = xs[1] - xs[0];
          auto x2x0 = xs[2] - xs[0];
          A(0, 0) = x1x0.l2NormSqr();
          A(1, 0) = A(0, 1) = x1x0.dot(x2x0);
          A(1, 1) = x2x0.l2NormSqr();

          auto IA = inverse(A);
          auto lnJ = zs::log(determinant(A) * determinant(IB)) / 2;
          E = dt * dt * vole *
              (model.mu / 2 * (trace(IB * A) - 2 - 2 * lnJ) +
               model.lam / 2 * lnJ * lnJ);
        }
        atomic_add(exec_cuda, &res[0], E);
      });
      // contacts
      {
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
#if 1
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // projection
      pol(zs::range(verts.size()),
          [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
           tag] ZS_LAMBDA(int vi) mutable {
            if (verts("x", 1, vi) > 0.8)
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
        auto BCorder = reinterpret_bits<int>(verts("BCorder", vi));
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
              dtiles_t &etemp, T dt, const ZenoConstitutiveModel &models)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp},
          tempPB{verts.get_allocator(), {{"H", 9}}, verts.size()}, dt{dt},
          models{models} {}

    const tiles_t &verts;
    const tiles_t &eles;
    dtiles_t &vtemp;
    dtiles_t &etemp;

    // boundary contacts
    dtiles_t tempPB;
    // end contacts
    T dt;
    const ZenoConstitutiveModel &models;
  };

  template <typename Model>
  static void computeElasticGradientAndHessian(
      zs::CudaExecutionPolicy &cudaPol, const Model &model,
      const tiles_t &verts, const tiles_t &eles, dtiles_t &vtemp,
      dtiles_t &etemp, float dt, const zs::SmallString &gTag = "grad") {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                     etemp = proxy<space>({}, etemp),
                                     verts = proxy<space>({}, verts),
                                     eles = proxy<space>({}, eles), model, gTag,
                                     dt] __device__(int ei) mutable {
      auto IB = eles.template pack<2, 2>("IB", ei);
      auto inds =
          eles.template pack<3>("inds", ei).template reinterpret_bits<int>();
      auto vole = eles("vol", ei);
      vec3 xs[3] = {vtemp.template pack<3>("xn", inds[0]),
                    vtemp.template pack<3>("xn", inds[1]),
                    vtemp.template pack<3>("xn", inds[2])};
      mat2 A{}, temp{};
      auto dA_div_dx = zs::vec<T, 4, 9>::zeros();
      auto x1x0 = xs[1] - xs[0];
      auto x2x0 = xs[2] - xs[0];
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
      mat3 BCbasis[3];
      int BCorder[3];
      for (int i = 0; i != 3; ++i) {
        BCbasis[i] = verts.pack<3, 3>("BCbasis", inds[i]);
        BCorder[i] = reinterpret_bits<int>(verts("BCorder", inds[i]));
      }
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

      zs::vec<T, 9, 9> H;
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
      H *= dt * dt * vole;
      make_pd(H);

      // rotate and project
      for (int vi = 0; vi != 3; ++vi) {
        int offsetI = vi * 3;
        for (int vj = 0; vj != 3; ++vj) {
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
      etemp.tuple<9 * 9>("He", ei) = H;
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
                           {"gc", 3},
                           {"gE", 3},
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
    static dtiles_t etemp{eles.get_allocator(), {{"He", 9 * 9}}, eles.size()};

    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{verts, eles, vtemp, etemp, dt, models};

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

    /// optimizer

    for (int newtonIter = 0; newtonIter != 100; ++newtonIter) {
      // construct gradient, prepare hessian, prepare preconditioner
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P", i) = mat3::zeros();
              });
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               dt] __device__(int i) mutable {
                auto m = verts("m", i);
                auto v = verts.pack<3>("v", i);
                vtemp.tuple<3>("grad", i) =
                    m * vec3{0, -9, 0} * dt * dt -
                    m * (vtemp.pack<3>("xn", i) - vtemp.pack<3>("xtilde", i));
              });
#if 1
      match([&](auto &elasticModel) {
        computeElasticGradientAndHessian(cudaPol, elasticModel, verts, eles,
                                         vtemp, etemp, dt);
      })(models.getElasticModel());
#endif
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
                vtemp("P", 0, i) += m;
                vtemp("P", 4, i) += m;
                vtemp("P", 8, i) += m;
              });
      cudaPol(zs::range(eles.size()),
              [vtemp = proxy<space>({}, vtemp), etemp = proxy<space>({}, etemp),
               eles = proxy<space>({}, eles)] __device__(int ei) mutable {
                constexpr int dim = 3;
                auto inds = eles.pack<3>("inds", ei).reinterpret_bits<int>();
                auto He = etemp.pack<dim * 3, dim * 3>("He", ei);
                for (int vi = 0; vi != 3; ++vi) {
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
        for (; iter != 10000; ++iter) {
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
      if (res < 1e-7) {
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
      find_ground_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha);

      T E{E0};
#if 1
      do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });

        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn");
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

ZENDEFNODE(CodimStepping,
           {{"ZSParticles", "ZSBoundaryPrimitives", {"float", "dt", "0.01"}},
            {"ZSParticles"},
            {},
            {"FEM"}});

} // namespace zeno