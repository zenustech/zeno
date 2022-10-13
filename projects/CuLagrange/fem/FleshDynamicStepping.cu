#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
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

#include "../geometry/linear_system/mfcg.hpp"

// #include "collision_energy/vertex_face_collision.hpp"
#include "collision_energy/vertex_face_sqrt_collision.hpp"

namespace zeno {
    using T = float;
    using dtiles_t = zs::TileVector<T,32>;
    using tiles_t = typename ZenoParticles::particles_t;
    using vec3 = zs::vec<T, 3>;
    using mat3 = zs::vec<T, 3, 3>;

    // struct FEMBackwardEulerSystem {
    //     constexpr auto dFAdF(const mat3& A) {
    //         zs::vec<T,9,9> M{};
    //         M(0,0) = M(1,1) = M(2,2) = A(0,0);
    //         M(3,0) = M(4,1) = M(5,2) = A(0,1);
    //         M(6,0) = M(7,1) = M(8,2) = A(0,2);

    //         M(0,3) = M(1,4) = M(2,5) = A(1,0);
    //         M(3,3) = M(4,4) = M(5,5) = A(1,1);
    //         M(6,3) = M(7,4) = M(8,5) = A(1,2);

    //         M(0,6) = M(1,7) = M(2,8) = A(2,0);
    //         M(3,6) = M(4,7) = M(5,8) = A(2,1);
    //         M(6,6) = M(7,7) = M(8,8) = A(2,2);

    //         return M;        
    //     }

    //     template<typename Pol,typename Model>
    //     T be_energy(Pol& pol,const Model& model,const zs::SmallString& tag,dtiles_t& vtemp,dtiles_t& etemp) {

    //     }

    //     template <typename Pol, typename Model>
    //     T energy(Pol &pol, const Model &model, const zs::SmallString& tag, dtiles_t& vtemp,dtiles_t& etemp) {
    //         using namespace zs;
    //         constexpr auto space = execspace_e::cuda;
    //         Vector<T> res{verts.get_allocator(), 1};
    //         res.setVal(0);
    //         bool shouldSync = pol.shouldSync();
    //         pol.sync(true);
    //         //   elastic potential
    //         pol(range(eles.size()), [verts = proxy<space>({}, verts),
    //                                 eles = proxy<space>({}, eles),
    //                                 vtemp = proxy<space>({}, vtemp),
    //                                 etemp = proxy<space>({},etemp),
    //                                 res = proxy<space>(res), tag, model = model,volf = volf] 
    //                                 ZS_LAMBDA (int ei) mutable {
    //             auto DmInv = eles.template pack<3, 3>("IB", ei);
    //             auto inds = eles.template pack<4>("inds", ei).template reinterpret_bits<int>();
    //             vec3 xs[4] = {vtemp.pack<3>(tag, inds[0]), vtemp.pack<3>(tag, inds[1]),
    //                         vtemp.pack<3>(tag, inds[2]), vtemp.pack<3>(tag, inds[3])};
    //             mat3 FAct{};
    //             {
    //             auto x1x0 = xs[1] - xs[0];
    //             auto x2x0 = xs[2] - xs[0];
    //             auto x3x0 = xs[3] - xs[0];
    //             auto Ds = mat3{x1x0[0], x2x0[0], x3x0[0], x1x0[1], x2x0[1],
    //                             x3x0[1], x1x0[2], x2x0[2], x3x0[2]};
    //             FAct = Ds * DmInv;

    //             FAct = FAct * etemp.template pack<3,3>("ActInv",ei);

    //             //   if(ei == 0) {
    //             //     printf("FAct in energy : \n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n",
    //             //         (float)FAct(0,0),(float)FAct(0,1),(float)FAct(0,2),
    //             //         (float)FAct(1,0),(float)FAct(1,1),(float)FAct(1,2),
    //             //         (float)FAct(2,0),(float)FAct(2,1),(float)FAct(2,2));
    //             //   }
    //             }

    //             auto psi = model.psi(FAct);
    //             auto vole = eles("vol", ei);

    //             T gpsi = 0;
    //             for(int i = 0;i != 4;++i)
    //                 gpsi += (-volf.dot(xs[i])/4); 

    //             atomic_add(exec_cuda, &res[0], (T)(vole * (psi + gpsi)));
    //         });
    //     // Bone Driven Potential Energy
    //         T lambda = model.lam;
    //         T mu = model.mu;
    //         auto nmEmbedVerts = b_verts.size();
    //         if(b_bcws.size() != b_verts.size()){
    //             fmt::print("B_BCWS_SIZE = {}\t B_VERTS_SIZE = {}\n",b_bcws.size(),b_verts.size());
    //             throw std::runtime_error("B_BCWS SIZE AND B_VERTS SIZE NOT MATCH");
    //         }
    //         pol(range(nmEmbedVerts), [vtemp = proxy<space>({},vtemp),
    //             eles = proxy<space>({},eles),
    //             b_verts = proxy<space>({},b_verts),
    //             bcws = proxy<space>({},b_bcws),lambda,mu,tag,res = proxy<space>(res),bone_driven_weight = bone_driven_weight]
    //             ZS_LAMBDA(int vi) mutable {
    //                 auto ei = reinterpret_bits<int>(bcws("inds",vi));
    //                 if(ei < 0)
    //                     return;
    //                 auto inds = eles.pack<4>("inds",ei).reinterpret_bits<int>();
    //                 auto w = bcws.pack<4>("w",vi);

    //                 auto tpos = vec3::zeros();
    //                 for(size_t i = 0;i != 4;++i)
    //                     tpos += w[i] * vtemp.pack<3>(tag,inds[i]);
    //                 auto pdiff = tpos - b_verts.pack<3>("x",vi);

    //                 T stiffness = 2.0066 * mu + 1.0122 * lambda;
    //                 // if(eles("vol",ei) < 0)
    //                 //     printf("WARNING INVERT TET DETECTED<%d> %f\n",ei,(float)eles("vol",ei));
    //                 T bpsi = (0.5 * bcws("cnorm",vi) * stiffness * bone_driven_weight * eles("vol",ei)) * pdiff.l2NormSqr();
    //                     // bpsi = (0.5 * bcws("cnorm",vi) * lambda * bone_driven_weight) * pdiff.dot(pdiff);
    //                     // the cnorm here should be the allocated volume of point in embeded tet 
    //                 atomic_add(exec_cuda, &res[0], (T)bpsi);
    //         });
    //         pol.sync(shouldSync);
    //         return res.getVal();
    //     }        
    // };

};