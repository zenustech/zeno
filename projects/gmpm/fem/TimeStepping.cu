#include "../Structures.hpp"
#include "../Utils.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/geometry/PoissonDisk.hpp"
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
  using T = float;
  using dtiles_t = zs::TileVector<T, 32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;
  using mat3 = zs::vec<T, 3, 3>;

/// ref: codim-ipc
#if 0
  template <class T, int dim>
  void computeInversionFreeStepSize(zs::CudaExecutionPolicy &pol,
                                    const tiles_t &verts, const tiles_t &eles,
                                    const dtiles_t &searchDir, T &stepSize) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    zs::Vector<T> stepSizes{eles.get_allocator(), eles.size()};
    pol(zs::Collapse{eles.size()},
        [verts = proxy<space>({}, verts), eles = proxy<space>({}, eles),
         stepSizes = proxy<space>(stepSizes),
         searchDir = proxy<space>({}, searchDir)](int ei) {
          auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
          T x1 = verts("x", 0, inds[0]);
          T x2 = verts("x", 0, inds[1]);
          T x3 = verts("x", 0, inds[2]);
          T x4 = verts("x", 0, inds[3]);

          T y1 = verts("x", 1, inds[0]);
          T y2 = verts("x", 1, inds[1]);
          T y3 = verts("x", 1, inds[2]);
          T y4 = verts("x", 1, inds[3]);

          T z1 = verts("x", 2, inds[0]);
          T z2 = verts("x", 2, inds[1]);
          T z3 = verts("x", 2, inds[2]);
          T z4 = verts("x", 2, inds[3]);

          T p1 = searchDir("dir", 0, inds[0]);
          T p2 = searchDir("dir", 0, inds[1]);
          T p3 = searchDir("dir", 0, inds[2]);
          T p4 = searchDir("dir", 0, inds[3]);

          T q1 = searchDir("dir", 1, inds[0]);
          T q2 = searchDir("dir", 1, inds[1]);
          T q3 = searchDir("dir", 1, inds[2]);
          T q4 = searchDir("dir", 1, inds[3]);

          T r1 = searchDir("dir", 2, inds[0]);
          T r2 = searchDir("dir", 2, inds[1]);
          T r3 = searchDir("dir", 2, inds[2]);
          T r4 = searchDir("dir", 2, inds[3]);

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

          T t = getSmallestPositiveRealCubicRoot(a, b, c, d, (T)1.e-6);
          if (t >= 0)
            stepSizes[ei] = t;
          else
            stepSizes[ei] = limits<T>::max();
        });

    zs::Vector<T> res{eles.get_allocator(), 1};
    zs::reduce(pol, std::begin(stepSizes), std::end(stepSizes), std::begin(res),
               stepSize, zs::getmin<T>{});
    stepSize = res.getVal();
  }
#endif

  struct FEMSystem {
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
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model,
                               dt = this->dt] __device__(int ei) mutable {
        auto DmInv = eles.pack<3, 3>("IB", ei);
        auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
        vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
                      vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
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
            vtemp.tuple<3>(dstTag, vi) =
                vtemp.pack<3, 3>("P", vi) * vtemp.pack<3>(srcTag, vi);
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
      pol(range(numVerts),
          [execTag, vtemp = proxy<space>({}, vtemp), bTag] ZS_LAMBDA(
              int vi) mutable { vtemp.tuple<3>(bTag, vi) = vec3::zeros(); });
      // inertial
      pol(range(numVerts), [execTag, verts = proxy<space>({}, verts),
                            vtemp = proxy<space>({}, vtemp), dxTag, bTag,
                            dt = this->dt] ZS_LAMBDA(int vi) mutable {
        auto m = verts("m", vi);
        auto dx = vtemp.pack<3>(dxTag, vi);
        auto BCbasis = verts.pack<3, 3>("BCbasis", vi);
        dx = BCbasis.transpose() * m * BCbasis * dx;
        for (int d = 0; d != 3; ++d)
          atomic_add(execTag, &vtemp(bTag, d, vi), dx(d));
      });
      // elastic energy
      pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                           vtemp = proxy<space>({}, vtemp),
                           eles = proxy<space>({}, eles), dxTag, bTag,
                           dt = this->dt] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = 3;
        constexpr auto dimp1 = dim + 1;
        auto inds = eles.pack<dimp1>("inds", ei).reinterpret_bits<int>();
        zs::vec<T, dimp1 * dim> temp{};
        for (int vi = 0; vi != dimp1; ++vi)
          for (int d = 0; d != dim; ++d) {
            temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
          }
        auto He = etemp.pack<dim * dimp1, dim * dimp1>("He", ei);

        temp = He * temp;

        for (int vi = 0; vi != dimp1; ++vi)
          for (int d = 0; d != dim; ++d) {
            atomic_add(execTag, &vtemp(bTag, d, inds[vi]), temp[vi * dim + d]);
          }
      });
    }

    FEMSystem(const tiles_t &verts, const tiles_t &eles, dtiles_t &vtemp,
              dtiles_t &etemp, T dt)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp}, dt{dt} {}

    const tiles_t &verts;
    const tiles_t &eles;
    dtiles_t &vtemp;
    dtiles_t &etemp;
    T dt;
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
    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{verts, eles, vtemp, etemp, dt};

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

      // line search
      T alpha = 1.;

      find_intersection_free_stepsize(cudaPol, *zstets, vtemp, alpha, 2e-3f);

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0{};
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel, "xn0");
      })(models.getElasticModel());
      T E{E0};
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
    // getchar();

    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(ImplicitTimeStepping, {{"ZSParticles", {"float", "dt", "0.01"}},
                                  {"ZSParticles"},
                                  {},
                                  {"FEM"}});

} // namespace zeno