#include "../Structures.hpp"
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
struct QuasiStaticStepping : INode {
  using T = float;
  using dtiles_t = zs::TileVector<T,32>;
  using tiles_t = typename ZenoParticles::particles_t;
  using vec3 = zs::vec<T, 3>;
  using mat3 = zs::vec<T, 3, 3>;

  struct FEMSystem {
    template <typename Pol, typename Model>
    T energy(Pol &pol, const Model &model, const zs::SmallString tag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model] 
                               __device__(int ei) mutable {
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
        atomic_add(exec_cuda, &res[0], vole * psi);
      });
      return res.getVal();
    }
    template <typename Pol> void project(Pol &pol, const zs::SmallString tag) {
      using namespace zs;
      constexpr execspace_e space = execspace_e::cuda;
      // projection
      pol(zs::range(verts.size()),
          [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
           tag] ZS_LAMBDA(int vi) mutable {
            if (verts("x", 1, vi) > 0.5)
              vtemp.tuple<3>(tag, vi) = vec3::zeros();
          });
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
      // dx -> b
      pol(range(numVerts),
          [execTag, vtemp = proxy<space>({}, vtemp), bTag] ZS_LAMBDA(
              int vi) mutable { vtemp.tuple<3>(bTag, vi) = vec3::zeros(); });
      // elastic energy
      pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                           vtemp = proxy<space>({}, vtemp),
                           eles = proxy<space>({}, eles), dxTag, bTag] ZS_LAMBDA(int ei) mutable {
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
              dtiles_t &etemp)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp}{}

    const tiles_t &verts;
    const tiles_t &eles;
    dtiles_t &vtemp;
    dtiles_t &etemp;
    
  };


  T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
        const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<3>(tag0, pi);
              auto v1 = data.pack<3>(tag1, pi);
              atomic_add(exec_cuda, res.data(), v0.dot(v1));
            });
    return res.getVal();
  }
  T infNorm(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
            const zs::SmallString tag = "dir") {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res),
             tag] __device__(int pi) mutable {
              auto v = data.pack<3>(tag, pi);
              atomic_max(exec_cuda, res.data(), v.abs().max());
            });
    return res.getVal();
  }

  template <typename Model>
  void computeElasticGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                                        const Model& model,
                                        const tiles_t &verts,
                                        const tiles_t &eles,
                                        dtiles_t &vtemp,
                                        dtiles_t &etemp) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                     etemp = proxy<space>({}, etemp),
                                     verts = proxy<space>({}, verts),
                                     eles = proxy<space>({}, eles), model] __device__(int ei) mutable {
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
      auto vfdt = -vole * (dFdXT * vecP);

      for (int i = 0; i != 4; ++i) {
        auto vi = inds[i];
        for (int d = 0; d != 3; ++d)
          atomic_add(exec_cuda, &vtemp("grad", d, vi), vfdt(i * 3 + d));
      }

      auto Hq = model.first_piola_derivative(F, true_c);
      auto H = dFdXT * Hq * dFdX * vole;
      etemp.tuple<12 * 12>("He", ei) = H;
    });
  }

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto models = zstets->getModel();
    auto& verts = zstets->getParticles();
    auto& eles = zstets->getQuadraturePoints();

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

    FEMSystem A{verts,eles,vtemp,etemp};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    // use the previous simulation result as initial guess
    cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp), verts = proxy<space>({},verts)]
                  __device__(int i) mutable{
                auto x = verts.pack<3>("x",i);
                vtemp.tuple<3>("xtilde",i) = x;
    });

    for(int newtonIter = 0;newtonIter != 100;++newtonIter){
      match([&](auto &elasticModel) {
        computeElasticGradientAndHessian(cudaPol, elasticModel, verts, eles,
                                         vtemp, etemp);
      })(models.getElasticModel());

      cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts)]
              __device__(int i) mutable {
                vtemp.tuple<3>("grad",i) = vec3{0,0,0};
      });

  //  Prepare Preconditioning
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
                [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
                  __device__(int ei) mutable {
                    constexpr int dim = 3;
                    constexpr auto dimp1 = dim + 1;
                    auto inds = 
                        eles.pack<dimp1>("inds",ei).reinterpret_bits<int>();
                    auto He = etemp.pack<dim * dimp1,dim * dimp1>("He",ei);
                    for (int vi = 0; vi != dimp1; ++vi) {
                      for (int i = 3; i != dim; ++i)
                        for (int j = 3; j != dim; ++j) {
                          atomic_add(exec_cuda, &vtemp("P", i * dim + j, inds[vi]),
                                    He(vi * dim + i, vi * dim + j));
                        }
                    }
      });   

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P",i) = inverse(vtemp.pack<3,3>("P",i));
      });

      // Solve equation using PCG
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
        T zTrk = dot(cudaPol,vtemp,"r","q");
        auto residualPreconditionedNorm = std::sqrt(zTrk);
        auto localTol = std::min(0.5 * residualPreconditionedNorm, 1e-7);
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
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0;
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
    
    }

    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto newX = vtemp.pack<3>("xn", vi);
              verts.tuple<3>("x", vi) = newX;
            });


    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(QuasiStaticStepping, {{"ZSParticles"},
                                  {"ZSParticles"},
                                  {},
                                  {"FEM"}});

}