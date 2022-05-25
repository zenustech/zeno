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
        constexpr T BONE_DRIVEN_WEIGHT = 10.;
        constexpr auto space = execspace_e::cuda;
        Vector<T> res{verts.get_allocator(), 1};
        res.setVal(0);
        // kinematic potential
        pol(range(eles.size()), [verts = proxy<space>({}, verts),
                                eles = proxy<space>({}, eles),
                                vtemp = proxy<space>({}, vtemp),
                                res = proxy<space>(res), tag, model = model,g = g] 
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
            // elastic potential
            auto psi = model.psi(F);
            auto vole = eles("vol", ei);
            // gravity potential
            T gpsi = 0;
            for(size_t i = 0;i < 4;++i)
                gpsi += - vole * g.dot(xs[i]) / 4; 
            atomic_add(exec_cuda, &res[0], vole * psi + gpsi);
        });
        // bone driven potential
        T lambda = model.lam;
        pol(zs::range(bcws.size()), [vtemp = proxy<space>({},vtemp),
                eles = proxy<space>({},eles),
                bverts = proxy<space>({},bverts),
                bcws = proxy<space>({},bcws),lambda,tag,res = proxy<space>(res)]
                ZS_LAMBDA (int vi) mutable {
                    auto ei = reinterpret_bits<int>(bcws("inds",vi));
                    auto inds = eles.pack<4>("inds",vi).reinterpret_bits<int>();
                    auto w = bcws.pack<4>("w",vi);
                    auto tpos = vec3::zeros();
                    for(size_t i = 0;i < 4;++i)
                        tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
                    auto pdiff = bverts.pack<3>("x",vi) - tpos;
                    // we should use character edge length here
                    auto bpsi = (0.5 * bcws("cedge",vi) * lambda * BONE_DRIVEN_WEIGHT) * pdiff.dot(pdiff);
                    atomic_add(exec_cuda,&res[0], (T)bpsi);
        });
        return res.getVal();
    }
    template <typename Model>
    void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,const Model& model, const zs::SmallString tag) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        constexpr T BONE_DRIVEN_WEIGHT = 10.;
        // kinematic potential's gradient and hessian
        cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                        etemp = proxy<space>({}, etemp),
                                        verts = proxy<space>({}, verts),
                                        eles = proxy<space>({}, eles), model,g = g,tag] ZS_LAMBDA (int ei) mutable {

            auto DmInv = eles.pack<3, 3>("IB", ei);
            auto dFdX = dFdXMatrix(DmInv);
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
            auto P = model.first_piola(F);
            auto vole = eles("vol", ei);
            auto vecP = flatten(P);
            auto dFdXT = dFdX.transpose();
            auto fe = -vole * (dFdXT * vecP);

            for (int i = 0; i != 4; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &vtemp("grad", d, vi), fe(i * 3 + d) + vole * g[i] / 4);
            }
            auto Hq = model.first_piola_derivative(F, true_c);
            auto H = dFdXT * Hq * dFdX * vole;
            etemp.tuple<12 * 12>("H", ei) = H;
        });
        T lambda = model.lam;
        cudaPol(zs::range(bcws.size()),
            [bcws = proxy<space>({},bcws),bverts = proxy<space>({},bverts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles),lambda,tag]
                ZS_LAMBDA (int vi) mutable {
                    constexpr int dim = 3;
                    constexpr auto dimp1 = dim + 1;
                    auto ei = reinterpret_bits<int>(bcws("inds",vi));
                    auto inds = eles.pack<dimp1>("inds",ei).reinterpret_bits<int>();
                    auto w = bcws.pack<dimp1>("w",vi);
                    auto tpos = vec3::zeros();
                    for(size_t i = 0;i < dimp1;++i)
                        tpos += w[i] * vtemp.pack<dim>(tag,inds[i]);
                    auto pdiff = tpos - bverts.pack<dim>("x",vi);
                    for(size_t i = 0;i < dimp1;++i){
                        auto tmp = - pdiff * (bcws("cedge",vi) * lambda * BONE_DRIVEN_WEIGHT * w[i]);
                        // vtemp.tuple<3>("grad",inds[i]] = vtemp.pack<3>("grad",inds[i]) - pdiff * bcws.pack<1>("area",vi) * stiffness * BONE_DRIVEN_WEIGHT * w[i];
                        for (int d = 0; d != dim; ++d)
                            atomic_add(exec_cuda, &vtemp("grad", d, inds[i]), tmp[d]);
                    }
                    for(size_t i = 0;i != dimp1;++i)
                        for(size_t j = 0;j != dimp1;++j){
                            auto alpha = lambda * BONE_DRIVEN_WEIGHT * w[i] * w[j] * bcws("cedge",vi);
                            for(size_t k = 0;k != dim;++k)
                                atomic_add(exec_cuda, &etemp("H", (i * dim + k) * dim*dimp1 + j * dim + k, ei), alpha);
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
                int vi) mutable { vtemp.tuple<3>(bTag, vi) = zs::vec<T,3>::zeros(); });
        // kinematic residual
        pol(range(numEles), [execTag, etemp = proxy<space>({}, etemp),
                            vtemp = proxy<space>({}, vtemp),
                            eles = proxy<space>({}, eles), dxTag, bTag] ZS_LAMBDA(int ei) mutable {
            constexpr int dim = 3;
            constexpr auto dimp1 = dim + 1;
            auto inds = eles.pack<dimp1>("inds", ei).reinterpret_bits<int>();
            zs::vec<T, 12> temp{};
            for (int vi = 0; vi != dimp1; ++vi)
            for (int d = 0; d != dim; ++d) {
                temp[vi * dim + d] = vtemp(dxTag, d, inds[vi]);
            }
            auto He = etemp.pack<dim * dimp1, dim * dimp1>("H", ei);
            temp = He * temp;
            for (int vi = 0; vi != dimp1; ++vi)
                for (int d = 0; d != dim; ++d)
                    atomic_add(execTag, &vtemp(bTag, d, inds[vi]), temp[vi * dim + d]);
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
              dtiles_t &etemp, const tiles_t &bcws,const tiles_t &bverts,vec3 g)
        : verts{verts}, eles{eles}, vtemp{vtemp}, etemp{etemp}, bcws{bcws},bverts{bverts},g{g}{}

    const tiles_t &verts;
    const tiles_t &eles;
    tiles_t &vtemp;
    tiles_t &etemp;

    vec3 g; 
    const dtiles_t &bcws;
    const dtiles_t &bverts;
  };

  T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
        const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    Vector<T> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] ZS_LAMBDA(int pi) mutable {
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
             tag] ZS_LAMBDA(int pi) mutable {
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
    auto btl_res = get_param<float>("btl_res");
    auto rtol = get_param<float>("rtol");
    auto max_cg_iters = get_param<int>("max_cg_iters");
    auto max_newton_iters = get_param<int>("max_newton_iters");
    auto models = zstets->getModel();
    auto& verts = zstets->getParticles();
    auto& eles = zstets->getQuadraturePoints();
    // input from a primtive object

    T eps = 1e-3;
    constexpr int dim = 3;
    constexpr auto dimp1 = dim + 1;

    static dtiles_t vtemp{verts.get_allocator(),
                          {{"grad", dim},
                           {"P", dim*dim},
                           {"dir", dim},
                           {"xn", dim},
                           {"xn0", dim},
                           {"xtilde", dim},
                           {"temp", dim},
                           {"r", dim},
                           {"p", dim},
                           {"q", dim}},
                          verts.size()};
    static dtiles_t etemp{eles.get_allocator(), {{"H", dim*dimp1 * dim*dimp1}}, eles.size()};
    vtemp.resize(verts.size());
    etemp.resize(eles.size());

    FEMSystem A{verts,eles,vtemp,etemp,(*zstets)["bcws"],zssurf->getParticles(),vec3::from_array(gravity) * models.density};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    // use the previous simulation result as initial guess
    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp),
             verts = proxy<space>({}, verts)] ZS_LAMBDA(int vi) mutable {
              auto x = verts.pack<3>("x", vi);
              vtemp.tuple<3>("xn", vi) = x;
            });


    for(int newtonIter = 0;newtonIter != max_newton_iters;++newtonIter){
      cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts)]
              ZS_LAMBDA(int i) mutable {
                vtemp.tuple<3>("grad",i) = zs::vec<T,3>::zeros();
      });

      match([&](auto &elasticModel) {
        A.computeGradientAndHessian(cudaPol, elasticModel, "xn");
      })(models.getElasticModel());

  //  Prepare Preconditioning
      cudaPol(zs::range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp),
            verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                vtemp.tuple<9>("P", vi) = zs::vec<T,3,3>::zeros();
      });
      cudaPol(zs::range(eles.size()),
                [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
                  ZS_LAMBDA (int ei) mutable {
                    auto inds = 
                        eles.template pack<dimp1>("inds",ei).template reinterpret_bits<int>();
                    auto He = etemp.pack<12,12>("H",ei);
                    for (int vi = 0; vi != 12; ++vi) {
                    #if 1
                      for (int i = 0; i != 3; ++i)
                        for (int j = 0; j != 3; ++j) {
                          atomic_add(exec_cuda, &vtemp("P", i * 3 + j, inds[vi]),
                                    He(vi * 3 + i, vi * 3 + j));
                        }
                    #else
                      for (int j = 0; j != 3; ++j) {
                          atomic_add(exec_cuda, &vtemp("P", j * 3 + j, inds[vi]),
                                    He(vi * 3 + j, vi * 3 + j));
                      }
                    #endif
                    }
      });

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple<9>("P",i) = inverse(vtemp.pack<3,3>("P",i));
      });

      // if the grad is too small, return the result

      // Solve equation using PCG
        A.project(cudaPol,"grad");
 #if 0     
      {
        // solve for A dir = grad;
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                  vtemp.tuple<3>("dir", i) = vec3::zeros();
                });
        A.multiply(cudaPol, "dir", "temp");
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                  vtemp.tuple<3>("r", i) =
                      vtemp.pack<3>("grad", i) - vtemp.pack<3>("temp", i);
                });
        A.project(cudaPol, "r");
        A.precondition(cudaPol, "r", "q");
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
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
            vtemp.tuple<dim>("p", vi) =
                vtemp.pack<dim>("q", vi) + beta * vtemp.pack<dim>("p", vi);
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
        std::vector<float> resvec(max_cg_iters);
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
            for(;iter != max_cg_iters;++iter) {
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
                    Vector<T> spd_test{etemp.size(), memsrc_e::um, 0};
                    cudaPol(zs::range(etemp.size()),
                        [etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles),spd_test = proxy<space>(spd_test)] ZS_LAMBDA(int ei) mutable {
                            auto He = etemp.pack<dim*dimp1,dim*dimp1>("H",ei);
                            auto dx = zs::vec<T,12>::zeros();
                            auto inds = eles.pack<dimp1>("inds",ei).reinterpret_bits<int>();
                            for(int i = 0;i < dim*dimp1;++i)
                                dx[i] = vtemp.pack<dim>("p",inds[i/dim])[i % dim];
                            spd_test[ei] = dx.dot(He * dx);
                    });

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
                        vtemp.tuple<dim>("dir",vi) = vtemp.pack<dim>("dir",vi) + alpha * vtemp.pack<dim>("p",vi);
                        vtemp.tuple<dim>("r",vi) = vtemp.pack<dim>("r",vi) - alpha * vtemp.pack<dim>("q",vi);
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
                            vtemp.tuple<dim>("r",vi) = vtemp.pack<dim>("grad",vi) - vtemp.pack<dim>("q",vi);
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
              [vtemp = proxy<space>({}, vtemp)] ZS_LAMBDA(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0;
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel,"xn0");
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
    //                                       alpha] ZS_LAMBDA(int i) mutable {
    //       vtemp.tuple<3>("xn", i) =
    //           vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
    //     });
    //     match([&](auto &elasticModel) {
    //       E = A.energy(cudaPol, elasticModel,"xn");
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
                                        alpha] ZS_LAMBDA(int i) mutable {
        vtemp.tuple<3>("xn", i) =
            vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
      });
    
    }

    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts)] ZS_LAMBDA(int vi) mutable {
              auto newX = vtemp.pack<3>("xn", vi);
              verts.tuple<3>("x", vi) = newX;
            });


    set_output("ZSParticles", std::move(zstets));
  }
};

ZENDEFNODE(QuasiStaticStepping, {{"ZSParticles","zssurf","gravity"},
                                  {"ZSParticles"},
                                  {{"float","armijo","0.1"},{"float","wolfe","0.9"},
                                    {"float","btl_res","0.0001"},{"int","max_newton_iters","5"},
                                    {"int","max_cg_iters","1000"},{"float","rtol","0.000001"}
                                    },
                                  {"FEM"}});

}