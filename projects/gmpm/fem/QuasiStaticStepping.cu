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
    T energy(Pol &pol, const Model &model,const zeno::vec<3,T>& g, const zs::SmallString tag) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
    //   elastic potential
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model] 
                               ZS_LAMBDA (int ei) mutable {
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
    // gravity potential
      pol(range(verts.size()),
            [verts = proxy<space>({},verts),vtemp = proxy<space>({},vtemp),res = proxy<space>(res),tag,g = vec3::from_array(g)]
            ZS_LAMBDA (int vi) mutable {
                auto m = verts("m",vi);
                auto v0 = vtemp.pack<3>(tag,vi);
                auto gpsi = -m * v0.dot(g); 
                atomic_add(exec_cuda, &res[0], gpsi);
      });
    // bone-driven potential
      T stiffness = model.lam;
      pol(zs::range(everts.size()), [vtemp = proxy<space>({},vtemp),
            eles = proxy<space>({},eles),
            everts = proxy<space>({},everts),
            bcws = proxy<space>({},bcws),stiffness, tag,res = proxy<space>(res)]
            ZS_LAMBDA (int vi) mutable {
                auto inds = bcws.pack<4>("inds",vi).reinterpret_bits<int>();
                auto w = bcws.pack<4>("w",vi);
                auto tpos = vec3::zeros();
                for(size_t i = 0;i < 4;++i)
                    tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
                auto pdiff = everts.pack<3>("x",vi) - tpos;
                // we should use character edge length here
                auto bpsi = 0.5 * pdiff.dot(pdiff) * bcws("area",vi) * stiffness * 10;
                atomic_add(exec_cuda,&res[0], (T)bpsi);
      });

      return res.getVal();
    }

    template <typename Model>
    void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                                            const Model& model,
                                            const zeno::vec<3,T>& g) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                        etemp = proxy<space>({}, etemp),
                                        verts = proxy<space>({}, verts),
                                        eles = proxy<space>({}, eles), model] ZS_LAMBDA (int ei) mutable {
            auto DmInv = eles.pack<3, 3>("IB", ei);
            auto dFdX = dFdXMatrix(DmInv);
            auto inds = eles.pack<4>("inds", ei).reinterpret_bits<int>();
            vec3 xs[4] = {vtemp.pack<3>("xn", inds[0]), vtemp.pack<3>("xn", inds[1]),
                            vtemp.pack<3>("xn", inds[2]), vtemp.pack<3>("xn", inds[3])};
            mat3 F{};
            //   {
                auto x1x0 = xs[1] - xs[0];
                auto x2x0 = xs[2] - xs[0];
                auto x3x0 = xs[3] - xs[0];
                auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
                            x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
                F = Ds * DmInv;
            //   }
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

        cudaPol(zs::range(verts.size()),[   vtemp = proxy<space>({},vtemp),
                                            verts = proxy<space>({},verts),
                                            g = vec3::from_array(g)] ZS_LAMBDA (int vi) mutable {
            auto m = verts("m",vi);
            vtemp.tuple<3>("grad",vi) = vtemp.pack<3>("grad",vi) + m * g;
        });

        T stiffness = model.lam;

        cudaPol(zs::range(bcws.size()),
            [bcws = proxy<space>({},bcws),everts = proxy<space>({},everts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),stiffness]
                ZS_LAMBDA (int vi) mutable {
                    auto ei = reinterpret_bits<int>(bcws("inds",vi));
                    auto inds = etemp.pack<4>("inds",ei).reinterpret_bits<int>();
                    auto w = bcws.pack<4>("w",vi);
                    auto tpos = vec3::zeros();
                    for(size_t i = 0;i < 4;++i)
                        tpos += w[i] * vtemp.pack<3>("xn",inds[i]);
                    auto pdiff = tpos - everts.pack<3>("x",vi);

                    for(size_t i = 0;i < 4;++i){
                        auto tmp = - pdiff * (bcws("area",vi) * stiffness * 10 * w[i]);
                        // vtemp.tuple<3>("grad",inds[i]] = vtemp.pack<3>("grad",inds[i]) - pdiff * bcws.pack<1>("area",vi) * stiffness * 10 * w[i];
                        for (int d = 0; d != 3; ++d)
                            atomic_add(exec_cuda, &vtemp("grad", d, inds[i]), tmp[d]);
                    }

                    for(size_t i = 0;i != 4;++i)
                        for(size_t j = 0;j != 4;++j){
                            auto alpha = stiffness * 10 * w[i] * w[j] * bcws("area",vi);
                            for(size_t k = 0;k != 3;++k)
                                // atomic_add(exec_cuda,&etemp.tuple<12,12>("He",ei,i*3 + k,j*3 + k),alpha);
                                atomic_add(exec_cuda, &etemp("He", (i * 3 + k) * 12 + j * 3 + k, ei), alpha);
                        }
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
            // vtemp.tuple<3>(dstTag, vi) = vtemp.pack<3>(srcTag, vi);
          });
    }


    FEMSystem(const tiles_t &verts, const tiles_t &eles, dtiles_t &vtemp,
              dtiles_t &etemp, const tiles_t &bcws,const tiles_t &everts)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp}, bcws{bcws},everts{everts}{}

    const tiles_t &verts;
    const tiles_t &eles;
    dtiles_t &vtemp;
    dtiles_t &etemp;

    const dtiles_t &bcws;
    const dtiles_t &everts;
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

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto zssurf = get_input<ZenoParticles>("zssurf");
    auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
    auto armijo = get_param<float>("armijo");
    auto curvature = get_param<float>("wolfe");
    auto cg_res = get_param<float>("cg_res");
    auto btl_res = get_param<float>("btl_res");
    auto rtol = get_param<float>("rtol");
    auto max_iterations = get_param<int>("max_iter");
    auto models = zstets->getModel();
    auto& verts = zstets->getParticles();
    auto& eles = zstets->getQuadraturePoints();
    // input from a primtive object
    auto& everts = zssurf->getParticles();

    T eps = 1e-3;

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

    FEMSystem A{verts,eles,vtemp,etemp,(*zstets)["bcws"],everts};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    // use the previous simulation result as initial guess
    cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp), verts = proxy<space>({},verts)]
                  __device__(int i) mutable{
                auto x = verts.pack<3>("x",i);
                vtemp.tuple<3>("xtilde",i) = x;
    });

    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp),
             verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto x = verts.pack<3>("x", vi);
              vtemp.tuple<3>("xn", vi) = x;
            });


    for(int newtonIter = 0;newtonIter != 5;++newtonIter){
      cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts)]
              __device__(int i) mutable {
                vtemp.tuple<3>("grad",i) = vec3{0,0,0};
      });

      match([&](auto &elasticModel) {
        A.computeGradientAndHessian(cudaPol, elasticModel, gravity);
      })(models.getElasticModel());


    //   cudaPol(zs::range(vtemp.size()),
    //     [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts), gravity = vec3::from_array(gravity)] ZS_LAMBDA (int vi) mutable {
    //         auto m = verts("m",vi);
    //         vtemp.tuple<3>("grad",vi) = vtemp.pack<3>("grad",vi) + m * gravity;

    //         // if(vi == 0){
    //         //     auto grad = vtemp.pack<3>("grad",vi);
    //         //     printf("GRAD : (%f,%f,%f) wit h m = %f\n",grad[0],grad[1],grad[2],m);
    //         // }
    //     });


  //  Prepare Preconditioning
      cudaPol(zs::range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp),
            verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                // auto m = verts("m", i);
                vtemp.tuple<9>("P", vi) = mat3::zeros();
                // vtemp("P", 0, i) = m;
                // vtemp("P", 4, i) = m;
                // vtemp("P", 8, i) = m;

                // if(vi == 1){
                //     auto gradNorm = verts.pack<3>("grad",vi).norm();
                //     printf("gradNorm[0] = (%f)\n",gradNorm);
                // }
      });

    

      cudaPol(zs::range(eles.size()),
                [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
                  ZS_LAMBDA (int ei) mutable {
                    constexpr int dim = 3;
                    constexpr auto dimp1 = dim + 1;
                    auto inds = 
                        eles.template pack<dimp1>("inds",ei).template reinterpret_bits<int>();
                    auto He = etemp.pack<dim * dimp1,dim * dimp1>("He",ei);
                    for (int vi = 0; vi != dimp1; ++vi) {
                    #if 1
                      for (int i = 0; i != dim; ++i)
                        for (int j = 0; j != dim; ++j) {
                          atomic_add(exec_cuda, &vtemp("P", i * dim + j, inds[vi]),
                                    He(vi * dim + i, vi * dim + j));
                        }
                    #else
                      for (int j = 0; j != dim; ++j) {
                          atomic_add(exec_cuda, &vtemp("P", j * dim + j, inds[vi]),
                                    He(vi * dim + j, vi * dim + j));
                      }
                    #endif
                    }
      });

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp)] __device__(int i) mutable {
                vtemp.tuple<9>("P",i) = inverse(vtemp.pack<3,3>("P",i));
      });

      // if the grad is too small, return the result

      // Solve equation using PCG
        A.project(cudaPol,"grad");
 #if 1     
      {
        // solve for A dir = grad;
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("dir", i) = vec3::zeros();
                });
        // {
        //     auto dirD = dot(cudaPol, vtemp, "dir", "dir");
        //     fmt::print("dir norm: {}\n", dirD);
        //     auto tmp = dot(cudaPol, vtemp, "grad", "grad");
        //     fmt::print("grad norm: {}\n", tmp);
        // }
        // temp = A * dir
        A.multiply(cudaPol, "dir", "temp");
        // auto AdNorm = dot(cudaPol,vtemp,"temp","temp");
        // fmt::print("AdNorm: {}\n",AdNorm);
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
        // auto localTol = std::min(0.5 * residualPreconditionedNorm, 1.0);
        auto localTol = 0.1 * residualPreconditionedNorm;
        int iter = 0;
        for (; iter != 1000; ++iter) {
          if (iter % 100 == 0)
            fmt::print("cg iter: {}, norm: {}\n", iter,
                        residualPreconditionedNorm);
          
          if (residualPreconditionedNorm <= localTol){
            // T dg = dot(cudaPol,vtemp,"grad","dir");
            // if(dg > 0)
                break;
          }
          A.multiply(cudaPol, "p", "temp");
          A.project(cudaPol, "temp");

          T pAp = dot(cudaPol, vtemp, "temp", "p");
          if(pAp < 0){
                fmt::print("INVALID pAp = {}\n",pAp);
                throw std::runtime_error("INVALID PQ, THE SYSTEM MATRIX IS NOT SPD");    
          }
          T alpha = zTrk / pAp;
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
#else
    {
        // A Matlab Version of Preconditioned CG Method
        // Set the initial guess of Ax = b
        T btol = std::sqrt(dot(cudaPol,vtemp,"grad","grad")) * rtol;
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA (int vi) mutable {
            vtemp.tuple<3>("dir",vi) = vec3::zeros(); 
            });

        // history of residuals
        std::vector<float> resvec(max_iterations);
        std::fill(resvec.begin(),resvec.end(),0);

        // Compute an initial residual of the equation, earlly exit if the initial guess is already a good enough solution
        // r = b - Ax
        A.multiply(cudaPol,"dir","temp");
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                vtemp.tuple<3>("r",vi) = vtemp.pack<3>("grad",vi) - vtemp.pack<3>("temp",vi);
        });
        A.project(cudaPol,"r");

        T normr = std::sqrt(dot(cudaPol,vtemp,"r","r"));
        T normr_act = normr;
        // solve pcg
        if(normr_act > btol){
            T normrmin = normr;
            T rho = 1;
            int moresteps = 0;
            int maxsteps = 5;
            int maxstagsteps = 1;
            int iter = 0;
            int stag = 0;
            for(;iter != max_iterations;++iter) {
                // precondition the residual
                A.project(cudaPol,"r");
                A.precondition(cudaPol,"r","q");

                T rhol = rho;
                rho = dot(cudaPol,vtemp,"r","q");
                if((rho == 0.) || std::isnan(rho)){
                    throw std::runtime_error("INVALID RHO");
                }
                if (iter % 100 == 0){
                    T residualPreconditionedNorm = std::sqrt(rho);
                    fmt::print("cg iter: {}, norm: {}\n", iter,
                                residualPreconditionedNorm);
                }


                if(iter == 0)
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple<3>("p",vi) = vtemp.pack<3>("q",vi);
                    });// for the first iteration, choose preconditioned residual as the first conjugate direction
                else{
                    T beta = rho / rhol;
                    if((beta == 0 || std::isnan(beta))){
                        throw std::runtime_error("INVALID BETA");
                    }
                    // eval the current conjugate direction using p(k+1) = beta*p(k) + M*r(k+1)
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp),beta] ZS_LAMBDA(int vi) mutable {
                            vtemp.tuple<3>("p",vi) = vtemp.pack<3>("p",vi) * beta + vtemp.pack<3>("q",vi);
                    });
                }

                A.project(cudaPol,"p");
                // Eval the step pTAp
                A.multiply(cudaPol,"p","q");
                A.project(cudaPol,"q");
                T pq = dot(cudaPol,vtemp,"p","q");
                if(pq < 0 || std::isnan(pq)){
                    fmt::print("INVALID PQ = {}\n",pq);
                    Vector<T> spd_test{etemp.get_allocator(),etemp.size()};
                    cudaPol(zs::range(etemp.size()),
                        [etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) mutable {
                            auto He = etemp.pack<12,12>("he",ei);
                            auto dx = vec<T,12>::zeros();
                            auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                            for(int i = 0;i < 12;++i)
                                dx[i] = vtemp.pack<3>("p",inds[i/3])[i % 3];
                            spd_test[ei] = dx.dot(He * dx);
                        }
                    )

                    for(int i = 0;i < etemp.size();++i)
                        if(spd_test[i] < 0)
                            fmt::print("NON_SPD_TET<{}> : {}\n",i,spd_test[i]);

                    throw std::runtime_error("INVALID PQ, THE SYSTEM MATRIX IS NOT SPD");
                }
                
                T alpha = rho / pq;

                // check for stagnation
                T normp = std::sqrt(dot(cudaPol,vtemp,"p","p"));
                T normd = std::sqrt(dot(cudaPol,vtemp,"dir","dir"));
                if(normp*fabs(alpha) < eps * normd)
                    ++stag;
                else
                    stag = 0;
                // update residual and searching direction
                cudaPol(zs::range(vtemp.size()),
                    [vtemp = proxy<space>({},vtemp),alpha] ZS_LAMBDA(int vi) mutable {
                        vtemp.tuple<3>("dir",vi) = vtemp.pack<3>("dir",vi) + alpha * vtemp.pack<3>("p",vi);
                        vtemp.tuple<3>("r",vi) = vtemp.pack<3>("r",vi) - alpha * vtemp.pack<3>("q",vi);
                });

                normr = std::sqrt(dot(cudaPol,vtemp,"r","r"));
                normr_act = normr;
                resvec[iter] = normr;

                // check for convergence
                if(normr <= btol || stag >= maxstagsteps || moresteps){
                    // recalculate bnorm
                    A.multiply(cudaPol,"dir","q");
                    A.project(cudaPol,"q");
                    cudaPol(zs::range(vtemp.size()),
                        [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
                            vtemp.tuple<3>("r",vi) = vtemp.pack<3>("grad",vi) - vtemp.pack<3>("q",vi);
                        });
                    A.project(cudaPol,"r");
                    normr_act = std::sqrt(dot(cudaPol,vtemp,"r","r"));
                    resvec[iter + 1] = normr_act;
                    if(normr_act <= btol)
                        break;
                    else{
                        if(stag >= maxstagsteps && moresteps == 0)
                            stag = 0;
                        moresteps++;
                        if (moresteps >= maxsteps) {
                            fmt::print("PCG::tooSmallTolerence");
                            // throw std::runtime_error("PCG:tooSmallTolerence");
                            break;
                        }
                    }
                }

                if(normr_act < normrmin){
                    normrmin = normr_act;
                    // xmin = x;
                    // imin = iter;
                }
                if(stag >= maxstagsteps){
                    fmt::print("PCG_TERMINATE_DUE_TO STAGNATION : {}\n",iter);
                    break;
                }
                // A.multiply(cudaPol,"p","temp");
            }
        }
    
    }
#endif

    T res = infNorm(cudaPol, vtemp, "dir");
    if (res < 1e-4) {
        fmt::print("\t# newton optimizer ends in {} iters with residual {}\n",
                    newtonIter, res);
        break;
    }
    fmt::print("newton iter {}: direction residual {}, grad residual {}\n",
                newtonIter, res, infNorm(cudaPol, vtemp, "r"));

      // line search
      T alpha = 1.;
      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0;
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel,gravity, "xn0");
      })(models.getElasticModel());

    //   T dg = dot(cudaPol,vtemp,"grad","dir");
    //   if(fabs(dg) < btl_res)
    //     break;
    //   if(dg < 0){
    //       throw std::runtime_error("INVALID DESCENT DIRECTION");
    //   }

    //   dg = -dg;

    //   T E{E0};
    //   Backtracking Linesearch
    //   int max_line_search = 20;
    //   int line_search = 0;
    //   do {
    //     cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
    //                                       alpha] __device__(int i) mutable {
    //       vtemp.tuple<3>("xn", i) =
    //           vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
    //     });
    //     match([&](auto &elasticModel) {
    //       E = A.energy(cudaPol, elasticModel,gravity, "xn");
    //     })(models.getElasticModel());
    //     // fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
    //     fmt::print("Armijo : {} < {}\n",(E - E0)/alpha,dg);
    //     // test Armojo condition
    //     if (E - E0 < armijo * dg * alpha)
    //       break;
    //     alpha /= 2;
    //     ++line_search;
    //   } while (line_search < max_line_search);
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

ZENDEFNODE(QuasiStaticStepping, {{"ZSParticles","gravity"},
                                  {"ZSParticles"},
                                  {{"float","armijo","0.1"},{"float","wolfe","0.9"},
                                    {"float","cg_res","0.1"},{"float","btl_res","0.0001"},
                                    {"int","max_iter","1000"},{"float","rtol","0.000001"}
                                    },
                                  {"FEM"}});

}