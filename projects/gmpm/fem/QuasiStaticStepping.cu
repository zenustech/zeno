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
    T energy(Pol &pol, const Model &model, const zs::SmallString tag, dtiles_t& vtemp) {
      using namespace zs;
      constexpr auto space = execspace_e::cuda;
      Vector<T> res{verts.get_allocator(), 1};
      res.setVal(0);
    //   elastic potential
      pol(range(eles.size()), [verts = proxy<space>({}, verts),
                               eles = proxy<space>({}, eles),
                               vtemp = proxy<space>({}, vtemp),
                               res = proxy<space>(res), tag, model = model,volf = volf] 
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

        T gpsi = 0;
        for(int i = 0;i != 4;++i)
            gpsi += (-volf.dot(xs[i])/4); 

        atomic_add(exec_cuda, &res[0], (T)(vole * (psi + gpsi)));
      });
// Bone Driven Potential Energy
      T lambda = model.lam;
      T mu = model.mu;
      auto nmEmbedVerts = b_verts.size();
      if(b_bcws.size() != b_verts.size()){
          fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
          throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
      }
      pol(range(nmEmbedVerts), [vtemp = proxy<space>({},vtemp),
          eles = proxy<space>({},eles),
          b_verts = proxy<space>({},b_verts),
          bcws = proxy<space>({},b_bcws),lambda,mu,tag,res = proxy<space>(res),bone_driven_weight = bone_driven_weight]
          ZS_LAMBDA(int vi) mutable {
              auto ei = reinterpret_bits<int>(bcws("inds",vi));
              if(ei < 0)
                  return;
              auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
              auto w = bcws.pack<4>("w",vi);

              auto tpos = vec3::zeros();
              for(size_t i = 0;i != 4;++i)
                  tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
              auto pdiff = tpos - b_verts.pack<3>("x",vi);

              T stiffness = 2.0066 * mu + 1.0122 * lambda;
              // if(eles("vol",ei) < 0)
              //     printf("WARNING INVERT TET DETECTED<%d> %f\n",ei,(float)eles("vol",ei));
              T bpsi = (0.5 * bcws("cnorm",vi) * stiffness * bone_driven_weight * eles("vol",ei)) * pdiff.l2NormSqr();
                    // bpsi = (0.5 * bcws("cnorm",vi) * lambda * bone_driven_weight) * pdiff.dot(pdiff);
// the cnorm here should be the allocated volume of point in embeded tet 
              atomic_add(exec_cuda, &res[0], (T)bpsi);
      });

      return res.getVal();
    }
      // projection
    template <typename Pol> void project(Pol &pol, const zs::SmallString tag, dtiles_t& vtemp) {
    //   using namespace zs;
    //   constexpr execspace_e space = execspace_e::cuda;
    //   pol(zs::range(verts.size()),
    //       [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts),
    //        tag] ZS_LAMBDA(int vi) mutable {
    //         if (verts("x", 1, vi) > 0.5)
    //           vtemp.tuple<3>(tag, vi) = vec3::zeros();
    //       });
    }


    template <typename Model>
    void computeGradientAndHessian(zs::CudaExecutionPolicy& cudaPol,
                                            const Model& model,
                                            const zs::SmallString tag, 
                                            dtiles_t& vtemp,
                                            dtiles_t& etemp) {
        using namespace zs;
        constexpr auto space = execspace_e::cuda;
        // fmt::print("check here 0");
        cudaPol(zs::range(eles.size()), [vtemp = proxy<space>({}, vtemp),
                                        etemp = proxy<space>({}, etemp),
                                        bcws = proxy<space>({},b_bcws),
                                        b_verts = proxy<space>({},b_verts),
                                        verts = proxy<space>({}, verts),
                                        eles = proxy<space>({}, eles),tag, model, volf = volf] ZS_LAMBDA (int ei) mutable {
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
            auto vf = -vole * (dFdXT * vecP);

            auto mg = volf * vole / 4;
            for (int i = 0; i != 4; ++i) {
                auto vi = inds[i];
                for (int d = 0; d != 3; ++d)
                    atomic_add(exec_cuda, &vtemp("grad", d, vi), vf(i * 3 + d) + mg(d));
            }

            auto Hq = model.first_piola_derivative(F, true_c);
            auto H = dFdXT * Hq * dFdX * vole;

            etemp.tuple<12 * 12>("He", ei) = H;

            // if(ei == 0){
            //     printf("F : \n%f %f %f\n%f %f %f\n%f %f %f\n",
            //         F(0,0),F(0,1),F(0,2),
            //         F(1,0),F(1,1),F(1,2),
            //         F(2,0),F(2,1),F(2,2)
            //     );
            //     printf("ELM_H<%d>:%e %f %f\n",ei,Hq.norm(),(float)model.lam,(float)model.mu);
            // }
            // etemp.tuple<12 * 12>("Hec",ei) = H;
            // etemp.tuple<12 * 12>("Hec",ei) = H

                    // if(ei == 11221){
                    //     printf("H0:\n");
                    //     for(int i = 0;i != 12;++i)
                    //         printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                    //             (double)etemp.pack<12,12>("He",ei)(i,0),
                    //             (double)etemp.pack<12,12>("He",ei)(i,1),
                    //             (double)etemp.pack<12,12>("He",ei)(i,2),
                    //             (double)etemp.pack<12,12>("He",ei)(i,3),
                    //             (double)etemp.pack<12,12>("He",ei)(i,4),
                    //             (double)etemp.pack<12,12>("He",ei)(i,5),
                    //             (double)etemp.pack<12,12>("He",ei)(i,6),
                    //             (double)etemp.pack<12,12>("He",ei)(i,7),
                    //             (double)etemp.pack<12,12>("He",ei)(i,8),
                    //             (double)etemp.pack<12,12>("He",ei)(i,9),
                    //             (double)etemp.pack<12,12>("He",ei)(i,10),
                    //             (double)etemp.pack<12,12>("He",ei)(i,11)
                    //         );
                    // }


        });


        // cudaPol(zs::range(etemp.size()),
        //     [etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) {
        //             if(ei == 11221){
        //                 printf("H0:\n");
        //                 for(int i = 0;i != 12;++i)
        //                     printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        //                         (double)etemp.pack<12,12>("He",ei)(i,0),
        //                         (double)etemp.pack<12,12>("He",ei)(i,1),
        //                         (double)etemp.pack<12,12>("He",ei)(i,2),
        //                         (double)etemp.pack<12,12>("He",ei)(i,3),
        //                         (double)etemp.pack<12,12>("He",ei)(i,4),
        //                         (double)etemp.pack<12,12>("He",ei)(i,5),
        //                         (double)etemp.pack<12,12>("He",ei)(i,6),
        //                         (double)etemp.pack<12,12>("He",ei)(i,7),
        //                         (double)etemp.pack<12,12>("He",ei)(i,8),
        //                         (double)etemp.pack<12,12>("He",ei)(i,9),
        //                         (double)etemp.pack<12,12>("He",ei)(i,10),
        //                         (double)etemp.pack<12,12>("He",ei)(i,11)
        //                     );
        //                 printf("g0:\n");
        //                 auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
        //                 printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        //                     (double)vtemp("grad",0,inds[0]),
        //                     (double)vtemp("grad",1,inds[0]),
        //                     (double)vtemp("grad",2,inds[0]),
        //                     (double)vtemp("grad",0,inds[1]),
        //                     (double)vtemp("grad",1,inds[1]),
        //                     (double)vtemp("grad",2,inds[1]),
        //                     (double)vtemp("grad",0,inds[2]),
        //                     (double)vtemp("grad",1,inds[2]),
        //                     (double)vtemp("grad",2,inds[2]),
        //                     (double)vtemp("grad",0,inds[2]),
        //                     (double)vtemp("grad",1,inds[2]),
        //                     (double)vtemp("grad",2,inds[2])
        //                 );
        //             }
        // });


        // fmt::print("check here 1\n");
        T lambda = model.lam;
        T mu = model.mu;
        if(b_bcws.size() != b_verts.size()){
            fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
            throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
        }

        // fmt::print("check here 2\n");

        auto nmEmbedVerts = b_verts.size();
        cudaPol(zs::range(nmEmbedVerts),
            [bcws = proxy<space>({},b_bcws),b_verts = proxy<space>({},b_verts),vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),
                eles = proxy<space>({},eles),lambda,mu,tag,bone_driven_weight = bone_driven_weight] ZS_LAMBDA(int vi) mutable {
                    auto ei = reinterpret_bits<int>(bcws("inds",vi));
                    if(ei < 0)
                        return;
                    auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
                    auto w = bcws.pack<4>("w",vi);
                    auto tpos = vec3::zeros();
                    for(size_t i = 0;i != 4;++i)
                        tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
                    auto pdiff = tpos - b_verts.pack<3>("x",vi);

                    T stiffness = 2.0066 * mu + 1.0122 * lambda;
                    // if(ei == 11221)
                    //     printf("INTERPOLATE WEIGHT:%f %f %d\n",(float)(stiffness * bone_driven_weight),1./(float)bcws("cnorm",vi),vi);
                    // if(fabs(1./(float)bcws("cnorm",vi) > 21.0))
                    //     printf("FOUND ELM<%d> %f\n",ei,1./(float)bcws("cnorm",vi));

                    // auto nid = eles.pack<4>("inds",11221).reinterpret_bits<int>()[0];
                    // for(int i = 0;i != 4;++i){
                    //     if(inds[i] == nid){
                    //         auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                    //         printf("EMBED_VERT_ID:<%d> %f tpos: %f %f %f bpos: %f %f %f pdiff: %f %f %f add: %f %f %f\n",vi,(float)(-stiffness * bcws("cnorm",vi) * bone_driven_weight * eles("vol",ei) * w[i]),
                    //             b_verts.pack<3>("x",vi)[0],
                    //             b_verts.pack<3>("x",vi)[1],
                    //             b_verts.pack<3>("x",vi)[2],
                    //             tpos[0],tpos[1],tpos[2],
                    //             pdiff[0],pdiff[1],pdiff[2],
                    //             tmp[0],tmp[1],tmp[2]
                    //         );
                    //     }
                    // }

                    for(size_t i = 0;i != 4;++i){
                        auto tmp = pdiff * (-stiffness * bcws("cnorm",vi) * bone_driven_weight * w[i] * eles("vol",ei)); 
                        // tmp = pdiff * (-lambda * bcws("cnorm",vi) * bone_driven_weight * w[i]);
                        for(size_t d = 0;d != 3;++d)
                            atomic_add(exec_cuda,&vtemp("grad",d,inds[i]),(T)tmp[d]);
                    }
                    for(int i = 0;i != 4;++i)
                        for(int j = 0;j != 4;++j){
                            T alpha = stiffness * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi) * eles("vol",ei);
                            // alpha = lambda * bone_driven_weight * w[i] * w[j] * bcws("cnorm",vi);
                            // if(ei == 11221)
                            //   if(i == 3 && j == 3)
                            //     printf("alpha : %f\n",alpha);
                            for(int d = 0;d != 3;++d){
                                // etemp("He",(i * 3 + d) * 12 + j * 3 + d,ei) += alpha;
                                if(isnan(alpha)){
                                    printf("nan alpha<%d,%d,%d> %f %f %f %f %f\n",vi,i,j,(float)lambda,(float)bone_driven_weight,(float)w[i],(float)w[j],(float)bcws("cnorm",vi));
                                }
                                atomic_add(exec_cuda,&etemp("He",(i * 3 + d) * 12 + j * 3 + d,ei),alpha);
                            }
                        }

        });
        // cudaPol(zs::range(etemp.size()),
        //     [etemp = proxy<space>({},etemp),vtemp = proxy<space>({},vtemp),eles = proxy<space>({},eles)] ZS_LAMBDA(int ei) {
        //             if(ei == 11221){
        //                 printf("H1:\n");
        //                 for(int i = 0;i != 12;++i)
        //                     printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        //                         (double)etemp.pack<12,12>("He",ei)(i,0),
        //                         (double)etemp.pack<12,12>("He",ei)(i,1),
        //                         (double)etemp.pack<12,12>("He",ei)(i,2),
        //                         (double)etemp.pack<12,12>("He",ei)(i,3),
        //                         (double)etemp.pack<12,12>("He",ei)(i,4),
        //                         (double)etemp.pack<12,12>("He",ei)(i,5),
        //                         (double)etemp.pack<12,12>("He",ei)(i,6),
        //                         (double)etemp.pack<12,12>("He",ei)(i,7),
        //                         (double)etemp.pack<12,12>("He",ei)(i,8),
        //                         (double)etemp.pack<12,12>("He",ei)(i,9),
        //                         (double)etemp.pack<12,12>("He",ei)(i,10),
        //                         (double)etemp.pack<12,12>("He",ei)(i,11)
        //                     );
        //                 printf("g:\n");
        //                 auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
        //                 printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        //                     (double)vtemp("grad",0,inds[0]),
        //                     (double)vtemp("grad",1,inds[0]),
        //                     (double)vtemp("grad",2,inds[0]),
        //                     (double)vtemp("grad",0,inds[1]),
        //                     (double)vtemp("grad",1,inds[1]),
        //                     (double)vtemp("grad",2,inds[1]),
        //                     (double)vtemp("grad",0,inds[2]),
        //                     (double)vtemp("grad",1,inds[2]),
        //                     (double)vtemp("grad",2,inds[2]),
        //                     (double)vtemp("grad",0,inds[3]),
        //                     (double)vtemp("grad",1,inds[3]),
        //                     (double)vtemp("grad",2,inds[3])
        //                 );
        //             }
        // });



        // fmt::print("check here 3");
    }
    template <typename Pol>
    void precondition(Pol &pol, const zs::SmallString srcTag,
                      const zs::SmallString dstTag,dtiles_t& vtemp) {
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
    template <typename Pol>
    void multiply(Pol &pol, const zs::SmallString dxTag,
                  const zs::SmallString bTag,
                  dtiles_t& vtemp,
                  const dtiles_t& etemp) {
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

    FEMSystem(const tiles_t &verts, const tiles_t &eles, const tiles_t &b_bcws, const tiles_t& b_verts,T bone_driven_weight,vec3 volf)
        : verts{verts}, eles{eles}, b_bcws{b_bcws}, b_verts{b_verts}, bone_driven_weight{bone_driven_weight},volf{volf}{}

    const tiles_t &verts;
    const tiles_t &eles;
    const tiles_t &b_bcws;  // the barycentric interpolation of embeded bones 
    const tiles_t &b_verts; // the position of embeded bones

    T bone_driven_weight;
    vec3 volf;
  };

  template<int pack_dim = 3>
  T dot(zs::CudaExecutionPolicy &cudaPol, dtiles_t &vertData,
        const zs::SmallString tag0, const zs::SmallString tag1) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    #if 1
    Vector<T> res{vertData.get_allocator(), 1};
    res.setVal(0);
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<pack_dim>(tag0, pi);
              auto v1 = data.pack<pack_dim>(tag1, pi);
              atomic_add(exec_cuda, &res[0], (T)v0.dot(v1));
            });
    return (T)res.getVal();
    #else
    Vector<double> res{vertData.get_allocator(), vertData.size()};
    cudaPol(range(vertData.size()),
            [data = proxy<space>({}, vertData), res = proxy<space>(res), tag0,
             tag1] __device__(int pi) mutable {
              auto v0 = data.pack<pack_dim>(tag0, pi);
              auto v1 = data.pack<pack_dim>(tag1, pi);
              res[pi] = v0.dot(v1);
              // atomic_add(exec_cuda, &ret[0], v0.dot(v1));
            });
    Vector<double> ret{vertData.get_allocator(), 1};
    auto sid = cudaPol.getStreamid();
    auto procid = cudaPol.getProcid();
    auto &context = Cuda::context(procid);
    auto stream = (cudaStream_t)context.streamSpare(sid);
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, temp_bytes, res.data(), ret.data(),
                              vertData.size(), std::plus<double>{}, 0., stream);
    Vector<std::max_align_t> temp{vertData.get_allocator(),
                                  temp_bytes / sizeof(std::max_align_t) + 1};
    cub::DeviceReduce::Reduce(temp.data(), temp_bytes, res.data(), ret.data(),
                              vertData.size(), std::plus<double>{}, 0., stream);
    context.syncStreamSpare(sid);
    return (T)ret.getVal();
    #endif

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
  // T avgForceRes(zs::CudaExecutionPolicy &cudaPol,const tiles_t &verts, dtiles_t &vertData,const zs::SmallString tag,const zeno::vec<3,T>& g) {
  //       using namespace zs;
  //       constexpr auto space = execspace_e::cuda;
  //       Vector<T> res{vertData.get_allocator(), 1};
  //       res.setVal(0);
  //       T gn = vec3::from_array(g).norm();
  //       cudaPol(range(vertData.size()),[data = proxy<space>({},vertData),verts = proxy<space>({},verts),tag,gn,res = proxy<space>(res)]
  //               ZS_LAMBDA(int vi) mutable {
  //                   auto ag = data.pack<3>(tag,vi).norm()/verts("m",vi)/gn;
  //                   atomic_add(exec_cuda,res.data(),ag);
  //               });
  //       return res.getVal()/verts.size();
  // }

  void apply() override {
    using namespace zs;
    auto zstets = get_input<ZenoParticles>("ZSParticles");
    auto gravity = get_input<zeno::NumericObject>("gravity")->get<zeno::vec<3,T>>();
    auto armijo = get_param<float>("armijo");
    auto curvature = get_param<float>("wolfe");
    auto cg_res = get_param<float>("cg_res");
    auto btl_res = get_param<float>("btl_res");
    auto models = zstets->getModel();
    auto& verts = zstets->getParticles();
    auto& eles = zstets->getQuadraturePoints();
    auto zsbones = get_input<ZenoParticles>("driven_bones");
    auto tag = get_param<std::string>("driven_tag");
    auto bone_driven_weight = get_param<float>("bone_driven_weight");
    auto newton_res = get_param<float>("newton_res");

    auto volf = vec3::from_array(gravity * models.density);

    static dtiles_t vtemp{verts.get_allocator(),
                          {{"grad", 3},
                           {"P", 9},
                        //    {"Pc",9},
                           {"dir", 3},
                           {"xn", 3},
                           {"xn0", 3},
                           {"xtilde", 3},
                           {"temp", 3},
                           {"r", 3},
                           {"p", 3},
                           {"q", 3}},
                          verts.size()};
    static dtiles_t etemp{eles.get_allocator(), {{"He", 12 * 12}/*,{"Hec",12*12}*/}, eles.size()};
    vtemp.resize(verts.size());
    etemp.resize(eles.size());
    FEMSystem A{verts,eles,(*zstets)[tag],zsbones->getParticles(),bone_driven_weight,volf};

    constexpr auto space = execspace_e::cuda;
    auto cudaPol = cuda_exec();

    // use the previous simulation result as initial guess
    // cudaPol(zs::range(vtemp.size()),
    //           [vtemp = proxy<space>({},vtemp), verts = proxy<space>({},verts)]
    //               __device__(int i) mutable{
    //             auto x = verts.pack<3>("x",i);
    //             vtemp.tuple<3>("xtilde",i) = x;
    // });

    // use the initial guess if given
    if(verts.hasProperty("init_x")){
      cudaPol(zs::range(verts.size()),
              [vtemp = proxy<space>({}, vtemp),
              verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                auto x = verts.pack<3>("init_x", vi);
                vtemp.tuple<3>("xn", vi) = x;
              });      
    }else{// use the previous simulation result
      cudaPol(zs::range(verts.size()),
              [vtemp = proxy<space>({}, vtemp),
              verts = proxy<space>({}, verts)] __device__(int vi) mutable {
                auto x = verts.pack<3>("x", vi);
                vtemp.tuple<3>("xn", vi) = x;
              });
    }

    for(int newtonIter = 0;newtonIter != 1000;++newtonIter){
      cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({},vtemp),verts = proxy<space>({},verts)]
              __device__(int i) mutable {
                vtemp.tuple<3>("grad",i) = vec3{0,0,0};
      });
      // fmt::print("COMPUTE GRADIENT AND HESSIAN\n",newtonIter);
      // fmt::print("volf_density:{}\n",volf_density);
      match([&](auto &elasticModel) {
        A.computeGradientAndHessian(cudaPol, elasticModel,"xn",vtemp,etemp);
      })(models.getElasticModel());
      // fmt::print("FINISH COMPUTE HESSIAN\n");

      // if(newtonIter == 0){
      //   T e0;
      //   match([&](auto &elasticModel) {
      //     e0 = A.energy(cudaPol, elasticModel,"xn",vtemp);
      //   })(models.getElasticModel());
      //   fmt::print("initial energy {}\n",e0);
      // }

      // break;

    //  T Hn = dot<144>(cudaPol,etemp,"He","He");
    //  fmt::print("Hn:{}\n",Hn);

    //  fmt::print("prepare Preconditioner \n",newtonIter);
    //  Prepare Preconditioning
      cudaPol(zs::range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp),
            verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                vtemp.tuple<9>("P", vi) = mat3::zeros();
      });

      cudaPol(zs::range(eles.size()),
        [vtemp = proxy<space>({},vtemp),etemp = proxy<space>({},etemp),eles = proxy<space>({},eles)]
            ZS_LAMBDA (int ei) mutable {
            constexpr int dim = 3;
            constexpr auto dimp1 = dim + 1;
            auto inds = 
                eles.template pack<dimp1>("inds",ei).template reinterpret_bits<int>();
            auto He = etemp.pack<dim * dimp1,dim * dimp1>("He",ei);
            // auto Hec = etemp.pack<dim * dimp1,dim * dimp1>("Hec",ei);

            // if(ei == 4723){
            //     printf("ELE<4723>:%d %d %d %d\n",inds[0],inds[1],inds[2],inds[3]);
            // }

            for (int vi = 0; vi != dimp1; ++vi) {
            #if 1
                for (int i = 0; i != dim; ++i)
                for (int j = i; j != dim; ++j){ 
                    atomic_add(exec_cuda, &vtemp("P", i * dim + j, inds[vi]),He(vi * dim + i, vi * dim + j));
                //   atomic_add(exec_cuda, &vtemp("P", j * dim + i, inds[vi]),He(vi * dim + i, vi * dim + j));
                }
            #else
                for (int j = 0; j != dim; ++j) {
                    atomic_add(exec_cuda, &vtemp("P", j * dim + j, inds[vi]),
                            He(vi * dim + j, vi * dim + j));
                }
            #endif
            }
      });

      // make sure it is symmetric
      cudaPol(zs::range(vtemp.size()),
          [vtemp = proxy<space>({}, vtemp),
            verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
                constexpr int dim = 3;
                for (int i = 0; i != dim; ++i)
                    for (int j = i+1; j != dim; ++j){ 
                        vtemp("P", j * dim + i, vi) = vtemp("P", i * dim + j, vi);
                //   atomic_add(exec_cuda, &vtemp("P", j * dim + i, inds[vi]),He(vi * dim + i, vi * dim + j));
                }
      });


    //   cudaPol(zs::range(vtemp.size()),
    //       [vtemp = proxy<space>({}, vtemp),
    //         verts = proxy<space>({}, verts)] ZS_LAMBDA (int vi) mutable {
    //             vtemp.tuple<9>("Pc", vi) = vtemp.pack<3,3>("P",vi);
    //   });

    //   T Pn = dot<9>(cudaPol,vtemp,"P","P");
    //   fmt::print("P_n:{}\n",Pn);

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({},vtemp)] __device__(int vi) mutable {
                  // we need to use double-precision inverse here, when the P matrix is nearly singular or has very large coeffs
                vtemp.tuple<9>("P",vi) = inverse(vtemp.pack<3,3>("P",vi).cast<double>());
      });

        // cudaPol(zs::range(vtemp.size()),
        //     [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
        //     auto P = vtemp.pack<3,3>("P",vi);
        //     auto Pc = vtemp.pack<3,3>("Pc",vi);
        //     // if(vi == 4966){
        //         auto Pdet = zs::determinant(P);
        //         auto PCdet = zs::determinant(Pc);
        //         if(P(0,0) < 0 || P(1,1) < 0 || P(2,2) < 0 || isnan(Pdet)) {
        //                         printf("NON_SPD_P<%d> %f : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\nSDP_PC<%d> % f: \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",vi,(float)Pdet,
        //                             (float)P(0,0),(float)P(0,1),(float)P(0,2),(float)P(1,0),(float)P(1,1),(float)P(1,2),(float)P(2,0),(float)P(2,1),(float)P(2,2),vi,(float)PCdet,
        //                             (float)Pc(0,0),(float)Pc(0,1),(float)Pc(0,2),(float)Pc(1,0),(float)Pc(1,1),(float)Pc(1,2),(float)Pc(2,0),(float)Pc(2,1),(float)Pc(2,2));
        //         }
        //         // }
        //     });

    //   Pn = dot<9>(cudaPol,vtemp,"P","P");
    //   fmt::print("Piv_n:{}\n",Pn);

    //   fmt::print("Solve Ax = b using PCG \n",newtonIter);

      // if the grad is too small, return the result
      // Solve equation using PCG
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
        A.multiply(cudaPol, "dir", "temp",vtemp,etemp);
        // auto AdNorm = dot(cudaPol,vtemp,"temp","temp");
        // fmt::print("AdNorm: {}\n",AdNorm);
        // r = grad - temp
        cudaPol(zs::range(vtemp.size()),
                [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                  vtemp.tuple<3>("r", i) =
                      vtemp.pack<3>("grad", i) - vtemp.pack<3>("temp", i);
                });
        A.project(cudaPol, "r",vtemp);
        A.precondition(cudaPol, "r", "q",vtemp); // q has the unit of length
        cudaPol(zs::range(vtemp.size()),
            [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("p", i) = vtemp.pack<3>("q", i);
        });



        T zTrk = dot(cudaPol,vtemp,"r","q");

        if(std::isnan(zTrk)){
            T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
            T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
            T gn = std::sqrt(dot(cudaPol,vtemp,"grad","grad"));
            T Pn = std::sqrt(dot<9>(cudaPol,vtemp,"P","P"));

            fmt::print("At the beginning NAN zTrk Detected r: {} q: {}, gn:{} Pn:{}\n",rn,qn,gn,Pn);
            throw std::runtime_error("NAN zTrk");
        }

        if(zTrk < 1e-4){
            T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
            T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
            fmt::print("\t# newton optimizer ends in {} iters with zTrk {} and grad {}\n",
            newtonIter, zTrk, infNorm(cudaPol, vtemp, "grad"));
            break;
        }
        if(zTrk < 0){
            T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
            T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
            fmt::print("\t#Begin invalid zTrk found in {} iters with zTrk {} and r {} and q {}\n",
                newtonIter, zTrk, infNorm(cudaPol, vtemp, "grad"),rn,qn);

            // cudaPol(zs::range(vtemp.size()),
            //     [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
            //         auto P = vtemp.pack<3,3>("P",vi);
            //         // auto Pc = vtemp.pack<3,3>("Pc",vi);
            //         auto PCdet = zs::determinant(Pc);
            //         // if(vi == 4966){
            //             auto Pdet = zs::determinant(P);
            //             if(P(0,0) < 0 || P(1,1) < 0 || P(2,2) < 0 || isnan(Pdet)) {
            //                     printf("CHECK NON_SPD_P<%d> %f : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\nSDP_PC<%d>: %f \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",vi,(float)Pdet,
            //                         (float)P(0,0),(float)P(0,1),(float)P(0,2),(float)P(1,0),(float)P(1,1),(float)P(1,2),(float)P(2,0),(float)P(2,1),(float)P(2,2),vi,(float)PCdet,
            //                         (float)Pc(0,0),(float)Pc(0,1),(float)Pc(0,2),(float)Pc(1,0),(float)Pc(1,1),(float)Pc(1,2),(float)Pc(2,0),(float)Pc(2,1),(float)Pc(2,2));
            //             }
            //     });

            throw std::runtime_error("INVALID zTrk");
        }
        auto residualPreconditionedNorm = std::sqrt(zTrk);
        // auto localTol = std::min(0.5 * residualPreconditionedNorm, 1.0);
        auto localTol = cg_res * residualPreconditionedNorm;
        // if(newtonIter < 10)
        //     localTol = 0.5 * residualPreconditionedNorm;
        int iter = 0;
        for (; iter != 1000; ++iter) {
          if (iter % 200 == 0)
            fmt::print("cg iter: {}, norm: {} zTrk: {} localTol: {}\n", iter, residualPreconditionedNorm,zTrk,localTol);
          
          if(zTrk < 0){
              T rn = std::sqrt(dot(cudaPol,vtemp,"r","r"));
              T qn = std::sqrt(dot(cudaPol,vtemp,"q","q"));
              fmt::print("\t# invalid zTrk found in {} iters with zTrk {} and r {} and q {}\n",
                  iter, zTrk,rn,qn);

              // fmt::print("FOUND NON_SPD P\n");
              // cudaPol(zs::range(vtemp.size()),
              //     [vtemp = proxy<space>({},vtemp)] ZS_LAMBDA(int vi){
              //         auto P = vtemp.pack<3,3>("P",vi);
              //         auto Pc = vtemp.pack<3,3>("Pc",vi);
              //         auto PCdet = zs::determinant(Pc);
              //         // if(vi == 4966){
              //             auto Pdet = zs::determinant(P);
              //             if(P(0,0) < 0 || P(1,1) < 0 || P(2,2) < 0 || isnan(Pdet)) {
              //                 printf("NON_SPD_P<%d> %f : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\nSDP_PC<%d>: %f \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",vi,(float)Pdet,
              //                     (float)P(0,0),(float)P(0,1),(float)P(0,2),(float)P(1,0),(float)P(1,1),(float)P(1,2),(float)P(2,0),(float)P(2,1),(float)P(2,2),vi,(float)PCdet,
              //                     (float)Pc(0,0),(float)Pc(0,1),(float)Pc(0,2),(float)Pc(1,0),(float)Pc(1,1),(float)Pc(1,2),(float)Pc(2,0),(float)Pc(2,1),(float)Pc(2,2));
              //             }
              //     });


              throw std::runtime_error("INVALID zTrk");
          }

          if (residualPreconditionedNorm <= localTol){ // this termination criterion is dimensionless
            // T dg = dot(cudaPol,vtemp,"grad","dir");
            // if(dg > 0)
                // fmt::print("finish with cg iter: {}, norm: {} zTrk: {}\n", iter,
                //             residualPreconditionedNorm,zTrk);
          
                break;
          }
          A.multiply(cudaPol, "p", "temp",vtemp,etemp);
          A.project(cudaPol, "temp",vtemp);

          T alpha = zTrk / dot(cudaPol, vtemp, "temp", "p");
          cudaPol(range(verts.size()), [verts = proxy<space>({}, verts),
                                        vtemp = proxy<space>({}, vtemp),
                                        alpha] ZS_LAMBDA(int vi) mutable {
            vtemp.tuple<3>("dir", vi) =
                vtemp.pack<3>("dir", vi) + alpha * vtemp.pack<3>("p", vi);
            vtemp.tuple<3>("r", vi) =
                vtemp.pack<3>("r", vi) - alpha * vtemp.pack<3>("temp", vi);
          });

          A.precondition(cudaPol, "r", "q",vtemp);
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
        fmt::print("FINISH SOLVING PCG with cg_iter = {}\n",iter);  
      }    
    // in case
      A.project(cudaPol,"dir",vtemp);
      A.project(cudaPol,"grad",vtemp);
      T res = infNorm(cudaPol, vtemp, "dir");// this norm is independent of descriterization

    //   fmt::print("NEWTON_ITER<{}> with gradn: {} and dirn: {}\n",newtonIter,gradn,res);

      if (res < newton_res) {
        // infNorm(cudaPol, vtemp, "grad")/(infNorm(cudaPol,eles,));
        fmt::print("\t# newton optimizer reach desired resolution in {} iters with residual {}\n",
                   newtonIter, res);
        break;
      }


      T dg = dot(cudaPol,vtemp,"grad","dir");
      if(fabs(dg) < btl_res){
        fmt::print("\t# newton optimizer reach stagnation point in {} iters with residual {}\n",
        newtonIter, res);
        break;
      }
      if(dg < 0){
          T gradn = std::sqrt(dot(cudaPol,vtemp,"grad","grad"));
          T dirn = std::sqrt(dot(cudaPol,vtemp,"dir","dir"));
          fmt::print("invalid dg = {} grad = {} dir = {}\n",dg);
          throw std::runtime_error("INVALID DESCENT DIRECTION");
      }

    //   fmt::print("DO LINE SEARCH\n");
      // line search
      T alpha = 1.;

      cudaPol(zs::range(vtemp.size()),
              [vtemp = proxy<space>({}, vtemp)] __device__(int i) mutable {
                vtemp.tuple<3>("xn0", i) = vtemp.pack<3>("xn", i);
              });
      T E0;
      match([&](auto &elasticModel) {
        E0 = A.energy(cudaPol, elasticModel, "xn0",vtemp);
      })(models.getElasticModel());


      dg = -dg;

      T E{E0};
    //   Backtracking Linesearch
      int max_line_search = 10;
      int line_search = 0;
      std::vector<T> armijo_buffer(max_line_search);
      do {
        cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                          alpha] __device__(int i) mutable {
          vtemp.tuple<3>("xn", i) =
              vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
        });
        match([&](auto &elasticModel) {
          E = A.energy(cudaPol, elasticModel, "xn",vtemp);
        })(models.getElasticModel());
        // fmt::print("E: {} at alpha {}. E0 {}\n", E, alpha, E0);
        // fmt::print("Armijo : {} < {}\n",(E - E0)/alpha,dg);
        armijo_buffer[line_search] = (E - E0)/alpha;
        // test Armojo condition
        if (((double)E - (double)E0) < (double)armijo * (double)dg * (double)alpha)
          break;
        alpha /= 2;
        ++line_search;
      } while (line_search < max_line_search);

    //   fmt::print("FINISH LINE SEARCH WITH LINE_SEARCH = {}\n",line_search);

      if(line_search == max_line_search){
          fmt::print("LINE_SEARCH_EXCEED: %f\n",dg);
          for(size_t i = 0;i != max_line_search;++i)
            fmt::print("AB[{}]\t = {} dg = {}\n",i,armijo_buffer[i],dg);
      }

      // fmt::print("FINISH NEWTON STEP WITH {} steps and line search {}\n",newtonIter,line_search);

      cudaPol(zs::range(vtemp.size()), [vtemp = proxy<space>({}, vtemp),
                                        alpha] __device__(int i) mutable {
        vtemp.tuple<3>("xn", i) =
            vtemp.pack<3>("xn0", i) + alpha * vtemp.pack<3>("dir", i);
      });
    
    }

    // T e1;
    // match([&](auto &elasticModel) {
    //   e1 = A.energy(cudaPol, elasticModel, "xn",vtemp);
    // })(models.getElasticModel());

    // fmt::print("finish energy {}\n",e1);


    cudaPol(zs::range(verts.size()),
            [vtemp = proxy<space>({}, vtemp), verts = proxy<space>({}, verts)] __device__(int vi) mutable {
              auto newX = vtemp.pack<3>("xn", vi);
              verts.tuple<3>("x", vi) = newX;
            });


    set_output("ZSParticles", zstets);
  }
};

ZENDEFNODE(QuasiStaticStepping, {{"ZSParticles","driven_bones","gravity"},
                                  {"ZSParticles"},
                                  {{"float","armijo","0.1"},{"float","wolfe","0.9"},
                                    {"float","cg_res","0.1"},{"float","btl_res","0.0001"},{"float","newton_res","0.001"},
                                    {"string","driven_tag","bone_bw"},{"float","bone_driven_weight","0.0"}},
                                  {"FEM"}});

}