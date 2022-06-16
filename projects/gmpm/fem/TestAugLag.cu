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

struct TestCodimStepping : INode {
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
  static constexpr T kappa0 = 1e3;
  inline static T kappa = kappa0;
  inline static vec3 extForce;

  static constexpr bool enable_elasticity = true;
  static constexpr bool enable_inertial = true;

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

  /// ref: codim-ipc
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
      pol(Collapse{numDofs},
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
      // auto res = infNorm(pol, vtemp, "cons");
      auto res = constraintResidual(pol);
      return res < 1e-2;
    }
    T checkDBCStatus(zs::CudaExecutionPolicy &pol) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      pol(Collapse{numDofs},
          [vtemp = proxy<space>({}, vtemp)] __device__(int vi) mutable {
            int BCorder = vtemp("BCorder", vi);
            if (BCorder > 0) {
              auto BCbasis = vtemp.pack<3, 3>("BCbasis", vi);
              auto BCtarget = vtemp.pack<3>("BCtarget", vi);
              auto cons = vtemp.pack<3>("cons", vi);
              auto xt = vtemp.pack<3>("xt", vi);
              auto x = vtemp.pack<3>("xn", vi);
              printf("%d-th vert (order [%d]): cur (%f, %f, %f) xt (%f, %f, %f)"
                     "\n\ttar(%f, %f, %f) cons (%f, %f, %f)\n",
                     vi, BCorder, (float)x[0], (float)x[1], (float)x[2],
                     (float)xt[0], (float)xt[1], (float)xt[2],
                     (float)BCtarget[0], (float)BCtarget[1], (float)BCtarget[2],
                     (float)cons[0], (float)cons[1], (float)cons[2]);
            }
          });
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
            T n = 0, d_ = 0;
            // https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf Eq5
            for (int d = 0; d != BCorder; ++d) {
              n += zs::sqr(cons[d]);
              d_ += zs::sqr(col(BCbasis, d).dot(xt) - BCtarget[d]);
            }
            num[vi] = n;
            den[vi] = d_;
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
    ///
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
                if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3) {
                  etemp.tuple<9 * 9>("He", ei) = H.zeros();
                  return;
                }
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
      if constexpr (enable_inertial) {
        pol(range(verts.size()),
            [verts = proxy<space>({}, verts), vtemp = proxy<space>({}, vtemp),
             res = proxy<space>(res), tag, extForce = extForce,
             dt = this->dt] __device__(int vi) mutable {
              // inertia
              auto m = verts("m", vi);
              auto x = vtemp.pack<3>(tag, vi);
              int BCorder = vtemp("BCorder", vi);
              if (BCorder != 3) {
                atomic_add(exec_cuda, &res[0],
                           (T)0.5 * m *
                               (x - vtemp.pack<3>("xtilde", vi)).l2NormSqr());
                // gravity
                atomic_add(exec_cuda, &res[0],
                           -m * extForce.dot(x - vtemp.pack<3>("xt", vi)) * dt *
                               dt);
              }
            });
      }
      if constexpr (enable_elasticity) {
        // elasticity
        pol(range(eles.size()),
            [verts = proxy<space>({}, verts), eles = proxy<space>({}, eles),
             vtemp = proxy<space>({}, vtemp), res = proxy<space>(res), tag,
             model = model, dt = this->dt] __device__(int ei) mutable {
              auto IB = eles.template pack<2, 2>("IB", ei);
              auto inds = eles.template pack<3>("inds", ei)
                              .template reinterpret_bits<int>();

              int BCorder[3];
              for (int i = 0; i != 3; ++i)
                BCorder[i] = vtemp("BCorder", inds[i]);
              if (BCorder[0] == 3 && BCorder[1] == 3 && BCorder[2] == 3) {
                return;
              }

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
      }
      // constraints
      if (includeAugLagEnergy) {
        computeConstraints(pol, tag);
        pol(range(numDofs), [vtemp = proxy<space>({}, vtemp),
                             res = proxy<space>(res),
                             kappa = kappa] __device__(int vi) mutable {
          // already updated during "xn" update
          auto cons = vtemp.template pack<3>("cons", vi);
          auto w = vtemp("ws", vi);
          auto lambda = vtemp.pack<3>("lambda", vi);
          atomic_add(
              exec_cuda, &res[0],
              (T)(-lambda.dot(cons) * w + 0.5 * kappa * w * cons.l2NormSqr()));
        });
      }
      return res.getVal();
    }
    template <typename Pol> void project(Pol &pol, const zs::SmallString tag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // projection
      pol(zs::range(numDofs),
          [vtemp = proxy<space>({}, vtemp), projectDBC = projectDBC,
           tag] ZS_LAMBDA(int vi) mutable {
            int BCfixed = vtemp("BCfixed", vi);
            if (projectDBC || (!projectDBC && BCfixed)) {
              int BCorder = vtemp("BCorder", vi);
              for (int d = 0; d != BCorder; ++d)
                vtemp(tag, d, vi) = 0;
            }
          });
    }
    template <typename Pol>
    void precondition(Pol &pol, const zs::SmallString srcTag,
                      const zs::SmallString dstTag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // precondition
      pol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp), srcTag,
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
      pol(range(numDofs), [execTag, vtemp = proxy<space>({}, vtemp),
                           bTag] ZS_LAMBDA(int vi) mutable {
        vtemp.template tuple<3>(bTag, vi) = vec3::zeros();
      });
      // inertial
      if constexpr (enable_inertial) {
        pol(range(numVerts),
            [execTag, verts = proxy<space>({}, verts),
             vtemp = proxy<space>({}, vtemp), dxTag, bTag,
             projectDBC = projectDBC] ZS_LAMBDA(int vi) mutable {
              auto m = verts("m", vi);
              auto dx = vtemp.template pack<3>(dxTag, vi);
              auto BCbasis = vtemp.template pack<3, 3>("BCbasis", vi);
              int BCorder = vtemp("BCorder", vi);
              int BCfixed = vtemp("BCfixed", vi);
              // dx = BCbasis.transpose() * m * BCbasis * dx;
              auto M = mat3::identity() * m;
              M = BCbasis.transpose() * M * BCbasis;
              if (projectDBC || (!projectDBC && BCfixed))
                for (int i = 0; i != BCorder; ++i)
                  for (int j = 0; j != BCorder; ++j)
                    M(i, j) = (i == j ? 1 : 0);
              dx = M * dx;
              for (int d = 0; d != 3; ++d)
                atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
            });
      }
      if constexpr (enable_elasticity) {
        // elasticity
        pol(range(numEles),
            [execTag, etemp = proxy<space>({}, etemp),
             vtemp = proxy<space>({}, vtemp), eles = proxy<space>({}, eles),
             dxTag, bTag] ZS_LAMBDA(int ei) mutable {
              constexpr int dim = 3;
              auto inds = eles.template pack<3>("inds", ei)
                              .template reinterpret_bits<int>();
              zs::vec<T, 3 * dim> temp{};
              for (int vi = 0; vi != 3; ++vi)
                for (int d = 0; d != dim; ++d) {
                  temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
                }
              auto He = etemp.template pack<dim * 3, dim * 3>("He", ei);

              temp = He * temp;

              for (int vi = 0; vi != 3; ++vi)
                for (int d = 0; d != dim; ++d) {
                  atomic_add(execTag, &vtemp(bTag, d, inds[vi]),
                             temp[vi * dim + d]);
                }
            });
      }
      // constraint hessian
      if (!BCsatisfied) {
        pol(range(numDofs), [execTag, vtemp = proxy<space>({}, vtemp), dxTag,
                             bTag, kappa = kappa] ZS_LAMBDA(int vi) mutable {
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
              const tiles_t &coVerts, const tiles_t &coEdges,
              const tiles_t &coEles, dtiles_t &vtemp, dtiles_t &etemp, T dt,
              const ZenoConstitutiveModel &models)
        : verts{verts}, edges{edges}, eles{eles}, coVerts{coVerts},
          coEdges{coEdges}, coEles{coEles}, vtemp{vtemp}, etemp{etemp},
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
    const tiles_t &coVerts, &coEdges, &coEles;
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

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    std::shared_ptr<ZenoParticles> zsboundary;
    if (has_input<ZenoParticles>("ZSBoundaryPrimitives"))
      zsboundary = get_input<ZenoParticles>("ZSBoundaryPrimitives");
    auto models = zstets->getModel();
    auto dt = get_input2<float>("dt");
    auto &verts = zstets->getParticles<true>();
    auto &edges = (*zstets)[ZenoParticles::s_surfEdgeTag];
    auto &eles = zstets->getQuadraturePoints();
    const tiles_t &coVerts =
        zsboundary ? zsboundary->getParticles() : tiles_t{};
    const tiles_t &coEdges =
        zsboundary ? (*zsboundary)[ZenoParticles::s_surfEdgeTag] : tiles_t{};
    const tiles_t &coEles =
        zsboundary ? zsboundary->getQuadraturePoints() : tiles_t{};

    auto coOffset = verts.size();
    auto numDofs = coOffset + coVerts.size();

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
         {"xt", 3},
         {"xn0", 3},
         {"xtilde", 3},
         {"temp", 3},
         {"r", 3},
         {"p", 3},
         {"q", 3}},
        numDofs};
    static dtiles_t etemp{eles.get_allocator(), {{"He", 9 * 9}}, eles.size()};

    vtemp.resize(numDofs);
    etemp.resize(eles.size());

    extForce = vec3{0, -9, 0};
    FEMSystem A{verts,  edges, eles,  coVerts, coEdges,
                coEles, vtemp, etemp, dt,      models};

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
      if (!BCsatisfied) {
        A.computeConstraints(cudaPol, "xn");
        auto cr = A.constraintResidual(cudaPol);
        if (A.areConstraintsSatisfied(cudaPol)) {
          // auto cr = A.constraintResidual(cudaPol);
          fmt::print("satisfied cons res [{}] at newton iter [{}]\n", cr,
                     newtonIter);
          // A.checkDBCStatus(cudaPol);
          getchar();
          projectDBC = true;
          BCsatisfied = true;
        }
        fmt::print(fg(fmt::color::alice_blue),
                   "newton iter {} cons residual: {}\n", newtonIter, cr);
      }

      // A.findCollisionConstraints(cudaPol, dHat, xi);
      auto [npp, npe, npt, nee, ncspt, ncsee] = A.getCnts();
      // construct gradient, prepare hessian, prepare preconditioner
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P", i) = mat3::zeros();
                vtemp.tuple<3>("grad", i) = vec3::zeros();
              });
      if constexpr (enable_inertial) {
        cudaPol(zs::range(verts.size()), [vtemp = proxy<space>({}, vtemp),
                                          verts = proxy<space>({}, verts),
                                          extForce = extForce,
                                          dt] __device__(int i) mutable {
          auto m = verts("m", i);
          auto v = verts.pack<3>("v", i);
          int BCorder = vtemp("BCorder", i);
          if (BCorder != 3)
            vtemp.tuple<3>("grad", i) =
                m * extForce * dt * dt -
                m * (vtemp.pack<3>("xn", i) - vtemp.pack<3>("xtilde", i));
        });
      }
      if constexpr (enable_elasticity) {
        match([&](auto &elasticModel) {
          A.computeElasticGradientAndHessian(cudaPol, elasticModel);
        })(models.getElasticModel());
      }
      // A.computeBoundaryBarrierGradientAndHessian(cudaPol);
      // A.computeBarrierGradientAndHessian(cudaPol);

      // rotate gradient and project
      cudaPol(zs::range(numDofs),
              [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
               projectDBC = projectDBC] __device__(int i) mutable {
                auto grad = vtemp.pack<3, 3>("BCbasis", i).transpose() *
                            vtemp.pack<3>("grad", i);
#if 0
                int BCfixed = vtemp("BCfixed", i);
                if (projectDBC || BCfixed == 1) {
                  if (int BCorder = vtemp("BCorder", i); BCorder > 0)
                    for (int d = 0; d != BCorder; ++d)
                      grad(d) = 0;
                }
#endif
                vtemp.tuple<3>("grad", i) = grad;
              });
      // apply constraints (augmented lagrangians) after rotation!
      if (!BCsatisfied) {
        // grad
        cudaPol(zs::range(numDofs), [vtemp = proxy<space>({}, vtemp),
                                     kappa = kappa] __device__(int i) mutable {
          // computed during the previous constraint residual check
          auto cons = vtemp.pack<3>("cons", i);
          auto w = vtemp("ws", i);
          vtemp.tuple<3>("grad", i) = vtemp.pack<3>("grad", i) +
                                      w * vtemp.pack<3>("lambda", i) -
                                      kappa * w * cons;
          for (int d = 0; d != 3; ++d)
            if (cons[d] != 0) {
              vtemp("P", 4 * d, i) += kappa * w;
            }
        });
        // hess (embedded in multiply)
      }

      // prepare preconditioner
      cudaPol(zs::range(verts.size()),
              [vtemp = proxy<space>({}, vtemp),
               verts = proxy<space>({}, verts)] __device__(int i) mutable {
                auto m = verts("m", i);
                int BCorder = vtemp("BCorder", i);
                int d = 0;
                for (; d != BCorder; ++d)
                  vtemp("P", d * 4, i) += m;
                for (; d != 3; ++d)
                  vtemp("P", d * 4, i) += m;
              });
      cudaPol(zs::range(numDofs),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P", i) = inverse(vtemp.pack<3, 3>("P", i));
              });

      // modify initial x so that it satisfied the constraint.

      // A dir = grad
      {
        // solve for A dir = grad;
        cudaPol(zs::range(numDofs),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("dir", i) = vec3::zeros();
                });
        // temp = A * dir
        A.multiply(cudaPol, "dir", "temp");
        // r = grad - temp
        cudaPol(zs::range(numDofs),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("r", i) =
                      vtemp.pack<3>("grad", i) - vtemp.pack<3>("temp", i);
                });
        A.project(cudaPol, "r");
        A.precondition(cudaPol, "r", "q");
        cudaPol(zs::range(numDofs),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("p", i) = vtemp.pack<3>("q", i);
                });
        T zTrk = dot(cudaPol, vtemp, "r", "q");
        auto residualPreconditionedNorm = std::sqrt(zTrk);
        auto localTol = 0.25 * residualPreconditionedNorm;
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
          cudaPol(range(numDofs), [vtemp = proxy<space>({}, vtemp),
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
          cudaPol(range(numDofs), [vtemp = proxy<space>({}, vtemp),
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
      if (res < 1e-2 && cons_res == 0) {
        fmt::print("\t# newton optimizer ends in {} iters with residual {}\n",
                   newtonIter, res);
        break;
      }

      fmt::print(fg(fmt::color::aquamarine),
                 "newton iter {}: direction residual {}, grad residual {}\n",
                 newtonIter, res, infNorm(cudaPol, vtemp, "grad"));

      // xn0 <- xn for line search
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0{};
      T cr0{};
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel, "xn0", !BCsatisfied);
      })(models.getElasticModel());
      cr0 = A.constraintResidual(cudaPol);

      // line search
      T alpha = 1.;
#if 0
      find_ground_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha);
      fmt::print("\tstepsize after ground: {}\n", alpha);
      A.intersectionFreeStepsize(cudaPol, xi, alpha);
      fmt::print("\tstepsize after intersection-free: {}\n", alpha);
      A.findCCDConstraints(cudaPol, alpha, xi);
      A.intersectionFreeStepsize(cudaPol, xi, alpha);
      fmt::print("\tstepsize after ccd: {}\n", alpha);
#endif

      T E{E0};
#if 1
      do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });

        // A.findCollisionConstraints(cudaPol, dHat, xi);
        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn", !BCsatisfied);
        })(models.getElasticModel());
        auto cr = A.constraintResidual(cudaPol);

        fmt::print("E: {} (cr: {}) at alpha {}. E0 {} (cr0: {})\n", E, cr,
                   alpha, E0, cr0);
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
          fmt::print("the heck, lambda updated!\n");
          getchar();
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

ZENDEFNODE(TestCodimStepping,
           {{"ZSParticles", "ZSBoundaryPrimitives", {"float", "dt", "0.01"}},
            {"ZSParticles"},
            {},
            {"FEM"}});

} // namespace zeno
